# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen3TTS model."""
# ======================================================================================
# 文件总览：Qwen3-TTS 主模型实现 (modeling_qwen3_tts.py)
# --------------------------------------------------------------------------------------
# 本文件实现 Qwen3-TTS 端到端语音合成大模型的核心 PyTorch 模型代码。
# 核心思路：把语音"分词"成离散编号（codes，共 16 层码本，每层 2048 个标准声音碎片），
# 然后用 LLM 自回归预测下一个编号来生成语音。
#
# 主模型 Qwen3TTSForConditionalGeneration 由三部分组成：
#   1. talker (Qwen3TTSTalkerForConditionalGeneration)：主 LLM（Qwen3 风格 Transformer）
#      - 自回归生成"第 0 层编号"（语音骨架）
#      - 内含 model (Qwen3TTSTalkerModel，Transformer 主体，含 codec_embedding/text_embedding)
#      - text_projection：把文字 embedding 投影到 talker 维度
#      - codec_head：线性层，预测第 0 层编号
#      - code_predictor (Qwen3TTSTalkerCodePredictorModelForConditionalGeneration)：小子 Transformer
#        在第 0 层编号基础上一次性预测第 1~15 层编号（多码本 LM / 多 token 预测 MTP）
#   2. speaker_encoder (Qwen3TTSSpeakerEncoder，仅 Base 模型)：ECAPA-TDNN
#      从参考音频提取说话人向量 x-vector，用于声音克隆
#   3. speech_tokenizer：语音 tokenizer（V1/V2），运行时注入，负责音频↔编号转换
#
# 模型类型 tts_model_type：
#   - base         ：声音克隆（ICL 上下文学习 + x-vector 说话人向量）
#   - custom_voice ：9 个预置音色
#   - voice_design ：自然语言描述生成音色
#
# 文件中模块的大致顺序：
#   1) 工具函数：download_weights_from_hf_specific、mel_spectrogram 等
#   2) 说话人编码器组件（ECAPA-TDNN 系列）：Res2NetBlock、SqueezeExcitationBlock、
#      AttentiveStatisticsPooling、TimeDelayNetBlock、SqueezeExcitationRes2NetBlock、
#      Qwen3TTSSpeakerEncoder
#   3) 基础 PreTrainedModel 类（Qwen3TTSPreTrainedModel / Qwen3TTSTalkerTextPreTrainedModel）
#   4) 位置编码 / 归一化 / 注意力相关公共组件：Qwen3TTSTalkerRotaryEmbedding、
#      Qwen3TTSRotaryEmbedding、Qwen3TTSRMSNorm、rotate_half、repeat_kv、
#      eager_attention_forward、apply_multimodal_rotary_pos_emb、apply_rotary_pos_emb
#   5) Talker 主体：Qwen3TTSTalkerAttention、Qwen3TTSTalkerResizeMLP、
#      Qwen3TTSTalkerTextMLP、Qwen3TTSTalkerDecoderLayer、Qwen3TTSTalkerModel、
#      Qwen3TTSTalkerForConditionalGeneration（含 forward_sub_talker_finetune、forward、
#      get_rope_index 等关键方法）
#   6) 子 talker / 编号预测器：Qwen3TTSTalkerCodePredictorModel、
#      Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
#   7) 主模型 Qwen3TTSForConditionalGeneration（含 from_pretrained、
#      extract_speaker_embedding、generate_speaker_prompt、generate_icl_prompt、generate）
# ======================================================================================

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import huggingface_hub
import torch
from huggingface_hub import snapshot_download
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (create_causal_mask,
                                        create_sliding_window_causal_mask)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast, ModelOutput)
from transformers.modeling_rope_utils import (ROPE_INIT_FUNCTIONS,
                                              dynamic_rope_update)
from transformers.modeling_utils import (ALL_ATTENTION_FUNCTIONS,
                                         PreTrainedModel)
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple, logging
from transformers.utils.hub import cached_file

from ...inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from .configuration_qwen3_tts import (Qwen3TTSConfig,
                                      Qwen3TTSSpeakerEncoderConfig,
                                      Qwen3TTSTalkerCodePredictorConfig,
                                      Qwen3TTSTalkerConfig)

logger = logging.get_logger(__name__)


# ======================================================================================
# 工具函数区
# --------------------------------------------------------------------------------------
# 这部分包含 HuggingFace Hub 权重下载、音频 mel 频谱计算等辅助函数，
# 供 from_pretrained 和 extract_speaker_embedding 调用。
# ======================================================================================

def download_weights_from_hf_specific(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Download model weights from Hugging Face Hub. Users can specify the
    allow_patterns to download only the necessary weights.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    assert len(allow_patterns) > 0
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE

    for allow_pattern in allow_patterns:
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_pattern,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_only,
        )
    return hf_folder


# ======================================================================================
# ECAPA-TDNN 说话人编码器组件
# --------------------------------------------------------------------------------------
# 下面这组类实现 ECAPA-TDNN（Emphasized Channel Attention, Propagation and Aggregation in
# TDNN）说话人特征提取网络，仅用于 tts_model_type == "base" 的声音克隆场景：
# 输入一段参考音频的 mel 频谱 → 输出定长的 x-vector（说话人向量）。
#
# 组合关系（数据流）：
#   mel 频谱
#     -> TimeDelayNetBlock (初始 TDNN)
#     -> [SqueezeExcitationRes2NetBlock] × 多层  (TDNN-Res2Net-TDNN-SE 残差块)
#     -> Multi-layer Feature Aggregation (MFA, 把各中间层特征拼接后再过一次 TDNN)
#     -> AttentiveStatisticsPooling       (注意力统计池化，把变长时间维压缩成定长向量)
#     -> 全连接                            (得到最终 speaker embedding)
# ======================================================================================

# Res2Net 残差块：把通道分成 scale 份，第 1 份直通，其余每份经过一个小 TDNN 后再与前一份相加，
# 用更少的参数构造出多尺度感受野，是 ECAPA-TDNN 的核心组件之一。
class Res2NetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TimeDelayNetBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, hidden_states):
        # 沿通道维把 hidden_states 切成 scale 份，第 0 份直通，第 1 份过一个 TDNN，
        # 后续每份 = TDNN(当前份 + 上一份输出)，把残差沿通道串起来。
        outputs = []
        for i, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        output = torch.cat(outputs, dim=1)
        return output


# Squeeze-and-Excitation (SE) 通道注意力块：在时间维上做全局平均得到一个通道权重向量，
# 再用它对原特征逐通道加权，让网络自己学每个通道的重要性。
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, se_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=se_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=se_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        # Squeeze: 在时间维做平均得到通道描述 -> Excitation: 两层 1x1 卷积 + relu/sigmoid 得到权重
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)

        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))

        # 把权重逐通道乘回原特征，实现"重要通道加强、次要通道抑制"
        return hidden_states * hidden_states_mean


# 注意力统计池化（Attentive Statistic Pooling）：
# 传统统计池化是无权重的均值/方差，这里先用一个注意力网络为每个时间步学一个权重，
# 再做加权的均值和标准差，把变长时间维压缩成定长向量（concat(mean, std)）。
# 这样既能聚合整段语音，又能让模型关注更重要的帧。
class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """

    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(
            in_channels=attention_channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def _length_to_mask(self, length, max_len=None, dtype=None, device=None):
        """Creates a binary mask for each sequence.

        Reference: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3

        Arguments
        ---------
        length : torch.LongTensor
            Containing the length of each sequence in the batch. Must be 1D.
        max_len : int
            Max length for the mask, also the size of the second dimension.
        dtype : torch.dtype, default: None
            The dtype of the generated mask.
        device: torch.device, default: None
            The device to put the mask variable.

        Returns
        -------
        mask : tensor
            The binary mask.
        """

        if max_len is None:
            max_len = length.max().long().item()  # using arange to generate mask
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        # 按权重 m 计算加权的均值和标准差
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states):
        # 输入 hidden_states 形状为 [batch, channel, time]
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)

        # Make binary mask of shape [N, 1, L]
        mask = self._length_to_mask(
            lengths * seq_length, max_len=seq_length, dtype=hidden_states.dtype, device=hidden_states.device
        )
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        total = mask.sum(dim=2, keepdim=True)

        # 第一步：先用朴素统计量作为上下文，让每帧看到全局信息
        mean, std = self._compute_statistics(hidden_states, mask / total)
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)

        # Apply layers
        # 用一个小 TDNN + tanh + 1x1 卷积算出每帧的注意力分数
        attention = self.conv(self.tanh(self.tdnn(attention)))

        # Filter out zero-paddings
        attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(attention, dim=2)
        # 用学到的注意力权重重新计算加权均值/标准差，最终拼成 [batch, 2*channel, 1]
        mean, std = self._compute_statistics(hidden_states, attention)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats

# TDNN 基础块：一层 1D 卷积（带 dilation 扩张感受野） + ReLU，
# 在 ECAPA-TDNN 里是最常用的"积木"。
class TimeDelayNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding="same",
            padding_mode="reflect",
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor):
        return self.activation(self.conv(hidden_states))

# ECAPA-TDNN 的核心残差块：TDNN -> Res2Net(多尺度) -> TDNN -> SE 通道注意力，
# 带跨层残差连接。一份"积木"重复堆叠 N 次就构成说话人编码器的主体。
class SqueezeExcitationRes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SqueezeExcitationBlock.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TimeDelayNetBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.res2net_block = Res2NetBlock(out_channels, out_channels, res2net_scale, kernel_size, dilation)
        self.tdnn2 = TimeDelayNetBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
        )
        self.se_block = SqueezeExcitationBlock(out_channels, se_channels, out_channels)

    def forward(self, hidden_state):
        residual = hidden_state

        # 顺序：1x1 TDNN 降维 -> Res2Net 多尺度 -> 1x1 TDNN -> SE 通道加权
        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)

        # 跨层残差，让深层的训练更稳定
        return hidden_state + residual


# 说话人编码器（仅 tts_model_type == "base" 时实例化）：
# 输入 mel 频谱 -> 输出 speaker embedding（x-vector）。
# 用于声音克隆：把"是谁的声音"浓缩成一个定长向量，喂给 talker。
class Qwen3TTSSpeakerEncoder(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://huggingface.co/papers/2005.07143).
    Use for Qwen3TTS extract speaker embedding.
    """

    def __init__(self, config: Qwen3TTSSpeakerEncoderConfig):
        super().__init__()
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(
            config.enc_dilations
        ):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have same length")
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        # 第 0 层：从 mel_dim 维映射到 enc_channels[0] 的初始 TDNN
        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )

        # SE-Res2Net layers
        # 中间若干层：每层都是 TDNN-Res2Net-TDNN-SE 残差块
        for i in range(1, len(config.enc_channels) - 1):
            self.blocks.append(
                SqueezeExcitationRes2NetBlock(
                    config.enc_channels[i - 1],
                    config.enc_channels[i],
                    res2net_scale=config.enc_res2net_scale,
                    se_channels=config.enc_se_channels,
                    kernel_size=config.enc_kernel_sizes[i],
                    dilation=config.enc_dilations[i],
                )
            )

        # Multi-layer feature aggregation
        # MFA：把中间各层特征在通道维拼接后，再用一次 TDNN 做融合
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        # Attentive Statistical Pooling
        # 注意力统计池化，把时间维压缩成定长向量
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        # Final linear transformation
        # 最终 1x1 卷积把特征维映射成 speaker embedding 维度 enc_dim
        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states):
        # 输入: mel 频谱 [batch, time, mel_dim] -> 转置成 [batch, mel_dim, time] 喂入卷积网络
        # Minimize transpose for efficiency
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states_list = []
        # 依次经过初始 TDNN 与各 SE-Res2Net 残差块，并保留中间各层特征供 MFA 聚合
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)

        # Multi-layer feature aggregation
        # 把第 1 层起的所有中间特征拼接后过 MFA
        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)

        # Attentive Statistical Pooling
        # 时间维变长 -> 定长向量
        hidden_states = self.asp(hidden_states)

        # Final linear transformation
        hidden_states = self.fc(hidden_states)

        # 去掉长度为 1 的时间维，得到 [batch, enc_dim] 的 speaker embedding
        hidden_states = hidden_states.squeeze(-1)
        return hidden_states


# 对 mel 频谱做对数压缩：log(clip(x, min=1e-5) * C)
# 把动态范围压窄，让数值更适合神经网络训练。
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

# 计算一段音频波形 y 的 mel 频谱图（声学特征常用形式）：
# 用 librosa 的 slaney 归一化 mel 滤波器组 + torch.stft 做短时傅里叶变换，
# 再做 mel 投影并取幅度谱，最后做对数压缩。
# 输出形状 [batch, num_mels, num_frames]，作为 speaker_encoder 的输入。
def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate the mel spectrogram of an input signal.
    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
        center (bool): Whether to pad the input to center the frames. Default is False.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device

    mel = librosa_mel_fn(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
    )

    mel_basis = torch.from_numpy(mel).float().to(device)
    hann_window = torch.hann_window(win_size).to(device)

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = dynamic_range_compression_torch(mel_spec)

    return mel_spec


# ======================================================================================
# 基础 PreTrainedModel 类
# --------------------------------------------------------------------------------------
# 两个基类分别给"主模型 Qwen3TTSForConditionalGeneration / 子 talker code_predictor"和
# "talker 主体（Qwen3TTSTalkerModel / Qwen3TTSTalkerForConditionalGeneration）"使用，
# 主要负责声明对各种 attention 后端、cache、gradient checkpointing 等能力的支持，
# 以及子模块权重初始化策略。
# ======================================================================================

# 主模型与 code_predictor 的基类。注：本移植版主要用于推理和微调，
# _init_weights 不是为从零训练准备的。
class Qwen3TTSPreTrainedModel(PreTrainedModel):
    config_class = Qwen3TTSConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3TTSDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False
    _supports_attention_backend = True

    def _init_weights(self, module):
        # important: this ported version of Qwen2.5OmniThinker isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


# talker 的基类，额外的 _supports_flex_attn / _supports_quantized_cache 表明 talker
# 支持 flex attention 和量化 KV 缓存，便于长序列推理时省显存。
class Qwen3TTSTalkerTextPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = False
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3TTSRMSNorm):
            module.weight.data.fill_(1.0)


# ======================================================================================
# 位置编码与归一化公共组件
# --------------------------------------------------------------------------------------
# 这里实现 Qwen3-TTS 用到的位置编码（RoPE）和归一化（RMSNorm）相关公共件。
#   - Qwen3TTSTalkerRotaryEmbedding：talker 用，3 段（时间/高/宽）多模态版 RoPE（mrope）
#   - Qwen3TTSRotaryEmbedding       ：code_predictor 子模型用，1D 版 RoPE
#   - Qwen3TTSRMSNorm               ：RMS 归一化（比 LayerNorm 省，不减均值）
#   - rotate_half / repeat_kv / eager_attention_forward / apply_*_rotary_pos_emb 是 attention 用的辅助函数
# ======================================================================================

# Talker 的 RoPE：多模态版本 mrope，把 cos/sin 沿时间、高、宽三个维度分别计算。
# 由于 TTS 输入只有文字和语音编号，没有图像，三段实际取相同的位置序号；
# 保留三段结构是为了与 Qwen3 多模态主干对齐，便于混用。
class Qwen3TTSTalkerRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3TTSThinkerText has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        # talker 的 mrope 把 inv_freq 扩展到 3 段（时间/高/宽），与多模态主干保持一致
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# code_predictor 子模型用的 1D RoPE：标准 1D 位置编码，与一般 LLM 一致。
class Qwen3TTSRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3TTSConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# RMSNorm：和 T5LayerNorm 等价，只做方差归一化（不减均值），再做可学习缩放。
# 比 LayerNorm 省算力，Qwen3 全网都用它。
@use_kernel_forward_from_hub("RMSNorm")
class Qwen3TTSRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3TTSRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 强制在 float32 下计算方差，避免半精度数值不稳；最后再转回原 dtype
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# 把张量后半部分旋转：[-x2, x1]，配合 cos/sin 实现 RoPE 旋转
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# GQA 分组查询注意力的 KV 复制：
# 把 key/value 从 (batch, num_kv_heads, seqlen, head_dim) 在 head 维度上重复 n_rep 次，
# 使其与 query 的 num_attention_heads 对齐。n_rep == 1 时直接返回原张量。
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# 纯 PyTorch 版（eager）的 attention 前向：标准 QK^T * scale + mask -> softmax -> * V。
# 当用户没有选 flash/sdpa 后端时使用；也是 ALL_ATTENTION_FUNCTIONS 里的兜底实现。
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # GQA: 把 key/value 复制到与 query 头数相同
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # 计算注意力分数 = QK^T / sqrt(head_dim)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # softmax 在 float32 上做以保证数值稳定，再转回 query dtype
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# 多模态版 RoPE（mrope）：把 q/k 的通道维按 mrope_section 切成时间/高/宽三段，
# 每段使用对应的 cos/sin 做旋转，让模型同时知道"在时间上、垂直方向、水平方向"上的位置。
# 文本部分三段位置序号相同，退化为普通 1D RoPE。
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, mrope_interleaved=False, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if mrope_interleaved:

        def apply_interleaved_rope(x, modality_num):
            x_t = x[0].clone()
            index_ranges = []
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                index_ranges.append((beg_idx, end_idx))
            for beg_idx, end_idx in index_ranges:
                x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
            return x_t

        dim = cos.shape[-1]
        modality_num = len(mrope_section)
        cos = torch.cat([apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(
            unsqueeze_dim
        )
    else:
        mrope_section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ======================================================================================
# 注意力、MLP 与解码层
# --------------------------------------------------------------------------------------
# Talker 用 Qwen3TTSTalkerAttention（mrope + 可选 sliding window），
# code_predictor 用 Qwen3TTSAttention（1D RoPE + 可选 sliding window）。
# MLP 上 talker 用带门控的 SwiGLU（Qwen3TTSTalkerTextMLP），
# 主模型还有 Qwen3TTSTalkerResizeMLP（普通两层 MLP）用于 text_projection。
# Decoder 层是 Attention + 残差 + RMSNorm + MLP + 残差 的标准结构。
# ======================================================================================

# Talker 的自注意力层（GQA + mrope）。
# - num_key_value_groups：每组 KV 共享的 query 头数（GQA 分组比例）
# - q_norm / k_norm：对每个 head_dim 做 RMSNorm，提升稳定性
# - sliding_window：可选滑动窗口注意力，只看局部窗口省算力（按层类型决定是否启用）
class Qwen3TTSTalkerAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3TTSRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3TTSRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        self.sliding_window = getattr(config, "sliding_window", None)
        self.rope_scaling = config.rope_scaling

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # 投影 + reshape 到 (batch, heads, seqlen, head_dim)，再做 RMSNorm
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 应用多模态 RoPE（mrope），把时间/高/宽三段位置信息编进 q、k
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"], self.rope_scaling["interleaved"]
        )

        if past_key_values is not None:
            # 把当前步的 K/V 写入 KV 缓存并取出（含历史 K/V），避免重复计算
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 优先用 flash / sdpa 后端，否则回退到 eager 实现
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        # 合并多头并做输出投影
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# 普通"升维-激活-降维"两层 MLP：用于 talker 的 text_projection，
# 把 text encoder 的 text_hidden_size 维映射到 talker 的 hidden_size 维。
class Qwen3TTSTalkerResizeMLP(nn.Module):
    def __init__(self, input_size: int, intermediate_size: int, output_size: int, act: str, bias=False):
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = ACT2FN[act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


# code_predictor 输出容器，多了一个 generation_steps 字段表示"已生成到第几层"。
@dataclass
class Qwen3TTSTalkerCodePredictorOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    generation_steps: Optional[int] = None


# SwiGLU 风格的 FFN（gate_proj 与 up_proj 同维，gate 路过激活再与 up 相乘，down_proj 降回 hidden_size）。
# 是 Qwen3 / Llama 系标准 MLP，talker / code_predictor 都用它。
class Qwen3TTSTalkerTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # down_proj(act(gate(x)) * up(x))，门控 + 升维 + 降维
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# 标量 1D 版 RoPE 应用：把 cos/sin 沿 unsqueeze_dim 升一维后，
# 对 q、k 应用 q*cos + rotate_half(q)*sin 的旋转（不区分多模态段）。
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# code_predictor 子模型的自注意力层（1D RoPE + 按层类型决定是否启用 sliding_window）。
# 与 Qwen3TTSTalkerAttention 的区别：用 1D RoPE，并且 sliding_window 由 layer_types 控制。
class Qwen3TTSAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3TTSConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3TTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3TTSRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        # 当 config.layer_types[layer_idx] 标记为 sliding_attention 时启用滑动窗口
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 应用 1D RoPE（无多模态段）
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # 把当前步 K/V 与缓存中历史 K/V 合并
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# code_predictor 用的解码层（标准 Pre-Norm Transformer Block）。
# 结构：input_layernorm -> Self-Attention -> 残差 -> post_attention_layernorm -> MLP -> 残差。
# attention_type 决定该层是全注意力还是滑动窗口注意力。
class Qwen3TTSDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3TTSConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3TTSAttention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3TTSTalkerTextMLP(config)
        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Pre-Norm 残差结构：先 norm 再 attention，输出与原 hidden 残差相加
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


# ======================================================================================
# 子 talker / 编号预测器 (Code Predictor)
# --------------------------------------------------------------------------------------
# 这是多码本 LM 的"小模型补 1~15 层"部分。它是一个小号 Transformer，
# 输入是 talker 第 0 层 hidden + 已生成编号的 embedding，输出对第 1~15 层编号的预测。
# 推理时通过多 token 预测（MTP）一次性补全每一时间步的剩余 15 层编号。
# 包含两层：
#   Qwen3TTSTalkerCodePredictorModel                    - 纯 backbone（decoder layers + norm + RoPE）
#   Qwen3TTSTalkerCodePredictorModelForConditionalGeneration - 在 backbone 之上加 lm_head，
#       支持 forward_finetune（一次出所有层 logits，训练用）与 forward（按 generation_steps
#       一次出某一层 logits，自回归推理用，配合 generate()）。
# ======================================================================================

# code_predictor 的 backbone：堆叠若干 Qwen3TTSDecoderLayer，外加最终 RMSNorm 与 1D RoPE。
# 注意：codec_embedding 是 ModuleList，每层编号（第 1~15 层）各有一份独立的 embedding 表。
class Qwen3TTSTalkerCodePredictorModel(Qwen3TTSPreTrainedModel):
    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor.model"

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, embedding_dim: int):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [Qwen3TTSDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        # 第 1 ~ num_code_groups-1 层的编号各自有一张 embedding 表，便于不同层使用不同语义空间
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, embedding_dim) for _ in range(config.num_code_groups - 1)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.codec_embedding

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        # 子模型只接受 inputs_embeds（已由上层把编号过 embedding 拼好），不接受 input_ids
        if input_ids is not None:
            raise ValueError("`input_ids` is expected to be `None`")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        # 构造 causal mask：全注意力层用 full_attention，滑动窗口层用 sliding_attention
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # 一次性算出所有层共享的 RoPE cos/sin，避免每层重复计算
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # 逐层前向，按层类型取对应的 mask（全注意力 / 滑动窗口）
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 最后过一次 RMSNorm
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# code_predictor 的条件生成外壳：在 backbone 之上加一组 lm_head（每个一层编号一个），
# 并提供 small_to_mtp_projection 把 talker 的 hidden 维（可能更大）映射到 code_predictor 维度。
# forward_finetune：训练用，一次并行预测所有 15 层编号（多 token 预测）。
# forward：推理用，按 generation_steps 单步生成某一层编号，配合自回归 generate()。
class Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(Qwen3TTSPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Qwen3TTSTalkerCodePredictorConfig
    base_model_prefix = "talker.code_predictor"

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig, talker_config: Qwen3TTSTalkerConfig):
        super().__init__(config)
        self.model = Qwen3TTSTalkerCodePredictorModel(config, talker_config.hidden_size)
        self.vocab_size = config.vocab_size
        # 15 个输出头，分别对应第 1~15 层编号的预测
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        # 维度不一致时把 talker 的 hidden 投影到 code_predictor 的 hidden；
        # 一致时用 Identity 直接透传，节省参数与算力。
        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = torch.nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = torch.nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # 训练用前向：把 [talker_hidden, 第 0 层编号 embed, 第 1 层编号 embed, ...] 拼成的
    # inputs_embeds 一次性送进 backbone，再用对应位置的 hidden 走 15 个 lm_head，
    # 同时并行得到所有 15 层编号的 logits，方便一次性计算多码本损失。
    def forward_finetune(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 先把 talker 维（如果不同）投影到 code_predictor 维
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # 第 i 层（1~num_code_groups-1）的预测 = lm_head[i-1](hidden_states 的第 i 个时间位置)
        # 一次性得到所有 15 层的 logits，stack 成 [batch, num_layers-1, seq, vocab]
        logits = []
        for i in range(1, self.config.num_code_groups):
            logits.append(self.lm_head[i-1](hidden_states[:, i]))
        logits = torch.stack(logits, dim=1)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return Qwen3TTSTalkerCodePredictorOutputWithPast(
            loss=loss,
            logits=logits
        )

    # 推理用前向（自回归 MTP）：根据当前 generation_steps 取对应层的输入 embedding 和 lm_head。
    # - Prefill 阶段（inputs_embeds.shape[1] > 1）：一次性"启动"，算出初始 generation_steps。
    # - Generation 阶段（每次 1 个 token）：用第 generation_steps-1 层的编号 embedding 作为输入，
    #   用第 generation_steps 个 lm_head 输出该层编号的预测。
    @can_return_tuple
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        generation_steps=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Prefill stage
        # 预填阶段：输入是 talker hidden 与第 0 层编号 embed 拼接而成，长度 = 层数 + 1
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2  # hidden & layer 0
        # Generation stage
        # 自回归阶段：根据当前 generation_steps 取出对应层的编号 embedding
        else:
            inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # 用第 generation_steps 个 lm_head 输出该层编号的预测
        logits = self.lm_head[generation_steps](hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return Qwen3TTSTalkerCodePredictorOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # 自增：下次进入 forward 时 generation_steps 指向下一层
            generation_steps=generation_steps + 1,
        )

    # 把 outputs.generation_steps 透传回 model_kwargs，供下一次 forward 使用
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs


# talker 输出容器，相比标准 CausalLMOutput 多了几个字段：
#   past_hidden          : 当前时间步最后一层 hidden，作为下一次 code_predictor 的输入
#   generation_step      : 当前生成到第几步（用于决定 trailing_text 的取位）
#   trailing_text_hidden : "尾部文字"的 hidden（流式模式下，文字与编号交错时的文字部分）
#   tts_pad_embed        : 文字对应的 pad 嵌入（用于对齐与填充）
@dataclass
class Qwen3TTSTalkerOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    past_hidden: Optional[torch.FloatTensor] = None
    generation_step: Optional[int] = None
    trailing_text_hidden: Optional[torch.FloatTensor] = None
    tts_pad_embed: Optional[torch.FloatTensor] = None


# talker 用的解码层（Pre-Norm 结构）。
# 与 Qwen3TTSDecoderLayer 类似，但用的是 talker 专用的 Qwen3TTSTalkerAttention（mrope），
# 不区分 sliding/full，统一全注意力。
class Qwen3TTSTalkerDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSTalkerAttention(config, layer_idx)

        self.mlp = Qwen3TTSTalkerTextMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


# ======================================================================================
# Talker 主体
# --------------------------------------------------------------------------------------
# Qwen3TTSTalkerModel：Transformer 主体（decoder layers + norm + mrope），
#   同时持有 codec_embedding（第 0 层编号的 embedding 表）和 text_embedding（文字编号的 embedding 表）。
# Qwen3TTSTalkerForConditionalGeneration：在 talker model 之上加 text_projection、codec_head
#   和 code_predictor，组成完整 talker。负责每步：
#   1) 调子 talker 生成第 1~15 层编号（MTP）；
#   2) 把 16 层编号 embed 求和（加 trailing_text_hidden）作为本步输入；
#   3) 跑 talker model 一步，codec_head 出第 0 层下一编号。
# ======================================================================================

# Talker 的 Transformer 主体：mrope + 多层 Qwen3TTSTalkerDecoderLayer + 最终 RMSNorm。
# 同时持有 codec_embedding（第 0 层编号 -> talker hidden 维）与 text_embedding（文字 -> text_hidden 维）。
class Qwen3TTSTalkerModel(Qwen3TTSTalkerTextPreTrainedModel):
    config_class = Qwen3TTSTalkerConfig
    base_model_prefix = "talker.model"

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [Qwen3TTSTalkerDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSTalkerRotaryEmbedding(config)
        self.gradient_checkpointing = False
        # 第 0 层编号的 embedding 表：编号 -> talker hidden 维
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # 文字编号的 embedding 表：编号 -> text_hidden 维（再由 talker.text_projection 映射到 hidden）
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.codec_embedding
    
    def get_text_embeddings(self):
        return self.text_embedding

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        # mrope 要求 position_ids 是 [3, batch, seq] 形状（时间/高/宽三段）。
        # 默认情况下三段使用同一个递增序列（对纯文本/编号场景足够）。
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # 如果传入 [4, batch, seq]，第 0 段是 text 的 position_ids（用于构造 mask），
        # 后 3 段才是 mrope 三段位置；否则从三段中取第 0 段作为 text position。
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # 一次性算出所有层共享的 mrope cos/sin
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# 完整 Talker：talker model + text_projection + codec_head + code_predictor。
# 是 Qwen3-TTS 的主 LLM：自回归生成第 0 层编号，并在每步调 code_predictor 补 1~15 层编号。
class Qwen3TTSTalkerForConditionalGeneration(Qwen3TTSTalkerTextPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Qwen3TTSTalkerConfig
    base_model_prefix = "talker"

    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__(config)
        self.model = Qwen3TTSTalkerModel(config)
        self.vocab_size = config.vocab_size
        # text_projection：把 text encoder 的 text_hidden_size 维映射到 talker hidden 维
        self.text_projection = Qwen3TTSTalkerResizeMLP(
            config.text_hidden_size, config.text_hidden_size, config.hidden_size, config.hidden_act, bias=True
        )

        # codec_head：talker 最后一层 hidden -> 第 0 层编号预测（线性层，无 bias）
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 子 talker：根据第 0 层 hidden + 已生成编号补 1~15 层编号
        self.code_predictor = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            config=config.code_predictor_config,
            talker_config=config
        )
        # rope_deltas：处理左 padding 时位置偏移量，由 get_rope_index 算出
        self.rope_deltas = None

        # Initialize weights and apply final processing
        self.post_init()

        # TODO: hack, modular cannot inherit multiple classes

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_text_embeddings(self):
        return self.model.get_text_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    # 子 talker 的训练前向封装：
    # 输入 codec_ids（每个时间步 16 层编号）和 talker_hidden_states（talker 输出的第 0 层 hidden）。
    # 把 [talker_hidden, 第0层编号embed, 第1层编号embed, ..., 第15层编号embed] 拼接成序列，
    # 调子 talker 一次性预测第 1~15 层编号，返回 logits 和 loss（多码本训练）。
    def forward_sub_talker_finetune(self, codec_ids, talker_hidden_states):
        assert len(codec_ids.shape) == 2
        assert len(talker_hidden_states.shape) == 2
        assert codec_ids.shape[0] == talker_hidden_states.shape[0]
        assert talker_hidden_states.shape[1] == self.config.hidden_size
        assert codec_ids.shape[1] == self.config.num_code_groups

        # 第 0 个位置：talker 第 0 层 hidden（带时间维，作为引导信号）
        sub_talker_inputs_embeds = [talker_hidden_states.unsqueeze(1)]

        # 后续每个位置：第 i 层编号对应的 embedding（第 0 层用 talker 的 codec_embedding，
        # 第 1~15 层用 code_predictor 的各自 embedding 表）
        for i in range(self.config.num_code_groups - 1):
            if i == 0:
                sub_talker_inputs_embeds.append(self.get_input_embeddings()(codec_ids[:, :1]))
            else:
                sub_talker_inputs_embeds.append(self.code_predictor.get_input_embeddings()[i-1](codec_ids[:, i:i+1]))
        sub_talker_inputs_embeds = torch.cat(sub_talker_inputs_embeds, dim=1)

        # 子 talker 一次出所有 15 层 logits，并用 codec_ids[:, 1:]（第 1~15 层真值）计算多码本损失
        sub_talker_outputs = self.code_predictor.forward_finetune(inputs_embeds=sub_talker_inputs_embeds,
                                                                 labels=codec_ids[:, 1:])

        sub_talker_logits = sub_talker_outputs.logits
        sub_talker_loss = sub_talker_outputs.loss
        return sub_talker_logits, sub_talker_loss

    @can_return_tuple
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        past_hidden=None,
        trailing_text_hidden=None,
        tts_pad_embed=None,
        generation_step=None,
        subtalker_dosample=None,
        subtalker_top_p=None,
        subtalker_top_k=None,
        subtalker_temperature=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        ```"""
        # Prefill：预填阶段。inputs_embeds 长度 > 1 表示一次性输入整段提示，不进入子 talker。
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_step = -1
            codec_ids = None
        # Generate：自回归生成阶段，每步只处理 1 个时间步。
        # 流程：先用子 talker 一次性补出第 1~15 层编号，再把 16 层 embed 求和 + trailing_text_hidden 作为本步输入。
        else:
            # 当前已生成的第 0 层编号 -> talker hidden 维 embedding
            last_id_hidden = self.get_input_embeddings()(input_ids)
            # 子 talker：以 [past_hidden, 当前第0层编号 embed] 作为输入，
            # 生成接下来 num_code_groups-1 = 15 个 token，对应第 1~15 层编号
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=subtalker_dosample,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            # 拼出完整 16 层编号：第 0 层（input_ids）+ 子 talker 生成的第 1~15 层
            codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
            # 把每层编号过对应 embedding 表，再在"层"维度求和，得到本时间步的输入 hidden
            # 这是多码本 LM 的标准做法：把多层信息压到一个输入向量
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i:i+1]) for i in range(self.config.num_code_groups - 1)],
                dim=1,
            )
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)

            # 流式模式下：如果还有未消费的"尾部文字"hidden，就加进去（让模型看到后面该读什么字）；
            # 否则用 tts_pad_embed 占位（语音段已经超过文字段）。
            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed
        # 处理左 padding 带来的位置偏移：在预填或第一步时，重新算 position_ids 和 rope_deltas；
        # 之后步骤直接用 cache_position + rope_deltas 推算，避免每步重算。
        if attention_mask is not None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    attention_mask,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                # 扩展到 3 段（时间/高/宽），mrope 需要这个形状
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # codec_head 出第 0 层下一时间步的编号预测（多码本的"骨架"由它生成）
        logits = self.codec_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)


        return Qwen3TTSTalkerOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            # hidden_states 既包含每层 hidden，也带上 codec_ids（16 层编号），供上层做后处理
            hidden_states=(outputs.hidden_states, codec_ids),
            attentions=outputs.attentions,
            # 当前时间步最后一层 hidden，作为下一次子 talker 的引导信号 past_hidden
            past_hidden=hidden_states[:, -1:, :],
            generation_step=generation_step + 1,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
        )

    # 计算 talker 在左 padding 场景下的 3D mrope 位置索引和偏移量。
    # 对纯文本/编号序列，时间/高/宽三段退化为同一个 1D 序号；这里基于 attention_mask 累加得到。
    # mrope_position_deltas = (max_position_ids + 1) - 有效长度，用于让后续 token 的位置连续递增。
    def get_rope_index(
        self,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        mrope_position_deltas = []

        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas

    # 把 talker 输出里的 past_hidden / generation_step / trailing_text_hidden / tts_pad_embed
    # 透传回 model_kwargs，供下一步自回归生成使用。
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["past_hidden"] = outputs.past_hidden
        model_kwargs["generation_step"] = outputs.generation_step
        model_kwargs["trailing_text_hidden"] = outputs.trailing_text_hidden
        model_kwargs["tts_pad_embed"] = outputs.tts_pad_embed
        return model_kwargs


# ======================================================================================
# 主模型 Qwen3TTSForConditionalGeneration
# --------------------------------------------------------------------------------------
# 整个 TTS 系统的对外入口。持有 talker（主 LLM）、可选的 speaker_encoder（仅 base 模型）、
# 以及运行时注入的 speech_tokenizer。提供：
#   - from_pretrained           ：从 HF 加载模型与语音 tokenizer
#   - extract_speaker_embedding ：从参考音频提 x-vector（仅 base）
#   - generate_speaker_prompt   ：把外部提好的 x-vector 整理成 talker 输入
#   - generate_icl_prompt        ：构造 ICL（上下文学习）提示，让模型模仿参考音频
#   - generate                   ：生成语音编号序列（batch 推理，左 padding 对齐）
# ======================================================================================
class Qwen3TTSForConditionalGeneration(Qwen3TTSPreTrainedModel, GenerationMixin):
    config_class = Qwen3TTSConfig

    def __init__(self, config: Qwen3TTSConfig):
        super().__init__(config)
        self.config = config

        self.talker = Qwen3TTSTalkerForConditionalGeneration(self.config.talker_config)

        if config.tts_model_type == "base":
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(self.config.speaker_encoder_config)
        else:
            self.speaker_encoder = None

        self.speech_tokenizer = None
        self.generate_config = None

        self.supported_speakers = self.config.talker_config.spk_id.keys()
        self.supported_languages = ["auto"]
        for language_id in self.config.talker_config.codec_language_id.keys():
            if "dialect" not in language_id:
                self.supported_languages.append(language_id)
        
        self.speaker_encoder_sample_rate = self.config.speaker_encoder_config.sample_rate
        self.tokenizer_type = self.config.tokenizer_type
        self.tts_model_size = self.config.tts_model_size
        self.tts_model_type = self.config.tts_model_type

        self.post_init()
    
    def load_speech_tokenizer(self, speech_tokenizer):
        self.speech_tokenizer = speech_tokenizer

    def load_generate_config(self, generate_config):
        self.generate_config = generate_config

    def get_supported_speakers(self):
        return self.supported_speakers

    def get_supported_languages(self):
        return self.supported_languages

    # 重写 from_pretrained：除加载主模型权重外，还要：
    #   1) 从 HF 拉取 speech_tokenizer 子目录的权重（运行时需要）；
    #   2) 用 Qwen3TTSTokenizer.from_pretrained 实例化并注入到 self.speech_tokenizer；
    #   3) 读取 generation_config.json 注入到 self.generate_config，供 generate() 使用。
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):
        # Hotfix to enable passing the correct attn implementation which is stored in the config but not in kwargs
        requested_attn_implementation = kwargs.pop("attn_implementation", None)
        if requested_attn_implementation is None and config and config._attn_implementation:
            requested_attn_implementation = config._attn_implementation

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            attn_implementation=requested_attn_implementation,
            **kwargs,
        )
        if not local_files_only and not os.path.isdir(pretrained_model_name_or_path):
            download_cache_dir = kwargs.get("cache_dir", cache_dir)
            download_revision = kwargs.get("revision", revision)
            download_weights_from_hf_specific(
                pretrained_model_name_or_path,
                cache_dir=download_cache_dir,
                allow_patterns=["speech_tokenizer/*"],
                revision=download_revision,
            )
        speech_tokenizer_path = cached_file(
            pretrained_model_name_or_path,
            "speech_tokenizer/config.json",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if speech_tokenizer_path is None:
            raise ValueError(f"""{pretrained_model_name_or_path}/{speech_tokenizer_path} not exists""")
        speech_tokenizer_dir = os.path.dirname(speech_tokenizer_path)
        speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_dir,
            *model_args,
            **kwargs,
        )
        model.load_speech_tokenizer(speech_tokenizer)

        generate_config_path = cached_file(
            pretrained_model_name_or_path,
            "generation_config.json",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        with open(generate_config_path, "r", encoding="utf-8") as f:
            generate_config = json.load(f)
        model.load_generate_config(generate_config)

        return model
    
    # 从一段 24kHz 参考音频提取 speaker embedding（x-vector）。
    # 流程：audio -> mel 频谱 -> ECAPA-TDNN speaker_encoder -> speaker embedding。
    # 仅 base 模型（带 speaker_encoder）可用，用于声音克隆。
    @torch.inference_mode()
    def extract_speaker_embedding(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000
        ).transpose(1, 2)
        speaker_embedding = self.speaker_encoder(mels.to(self.device).to(self.dtype))[0]
        return speaker_embedding

    # 把外部已经提好的参考说话人向量整理成 talker 可直接使用的列表。
    # 主要是搬到 talker 的 device / dtype 上；不做任何模型计算。
    @torch.inference_mode()
    def generate_speaker_prompt(
        self,
        voice_clone_prompt: list[dict]
    ):
        voice_clone_spk_embeds = []
        for index in range(len(voice_clone_prompt['ref_spk_embedding'])):
            ref_spk_embedding = voice_clone_prompt["ref_spk_embedding"][index].to(self.talker.device).to(self.talker.dtype)
            voice_clone_spk_embeds.append(ref_spk_embedding)

        return voice_clone_spk_embeds

    # 构造 ICL（上下文学习）提示：把"参考音频编号 ref_code"和"参考文字 ref_id + 待合成文字 text_id"
    # 拼成一段输入嵌入，让 talker 在生成时模仿参考音频的音色与风格。
    #
    # 输入：
    #   text_id   : 待合成文字的编号序列
    #   ref_id    : 参考音频对应的文字编号序列
    #   ref_code  : 参考音频的 16 层语音编号
    #   tts_*_embed : 控制标记（pad/eos）的嵌入
    #   non_streaming_mode : 非流式（文字与编号不对齐交错，整段文字在前）
    #
    # 输出：(icl_input_embed, trailing_text_hidden)
    #   - 非流式：text 段 + codec 段拼起来，trailing 全部用 pad
    #   - 流式：文字与编号按"较短的"对齐交错，多出来的文字作为 trailing（之后逐步喂给 talker）
    def generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ):
        # text embed (ref id + text id + eos) 1 T1 D
        # 文字嵌入 = text_projection(text_embedding([ref 文字编号, 待合成文字编号])) 后再拼一个 eos
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(torch.cat([ref_id, text_id],
                                                            dim=-1)))
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        # codec embed (codec bos + codec) 1 T2 D
        # 把 16 层参考编号分别过各自 embedding 表，在层维求和（与 talker 多码本输入一致），
        # 再前面拼一个 codec_bos 标记
        codec_embed = []
        for i in range(self.talker.config.num_code_groups):
            if i == 0:
                codec_embed.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(self.talker.code_predictor.get_input_embeddings()[i-1](ref_code[:, i:i+1]))
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat([self.talker.get_input_embeddings()(
                                    torch.tensor(
                                        [[
                                            self.config.talker_config.codec_bos_id,
                                        ]],
                                        device=self.talker.device,
                                        dtype=text_id.dtype,
                                    )
                                ), codec_embed], dim=1)
        # compute lens
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if non_streaming_mode:
            # 非流式：文字段全部"对齐到 codec_pad"（即文字位置只放文字，编号位置用 pad 占位），
            # 再拼上"编号段（codec 段每步对齐到 tts_pad）"，整段就是 ICL 提示
            icl_input_embed = text_embed + self.talker.get_input_embeddings()(
                                                torch.tensor(
                                                    [[
                                                        self.config.talker_config.codec_pad_id,
                                                    ] * text_lens],
                                                    device=self.talker.device,
                                                    dtype=text_id.dtype,
                                                )
                                            )
            icl_input_embed = torch.cat([icl_input_embed, codec_embed + tts_pad_embed], dim=1)
            # 非流式没有"尾部文字"，统一用 pad 占位
            return icl_input_embed, tts_pad_embed
        else:
            # 流式：文字与编号"交错对齐"。短的那段对齐到长的那段，剩下多的那部分作为 trailing。
            # 情况 A：文字比编号长 -> 前 codec_lens 个文字与编号对齐，剩余文字作为 trailing_text_hidden
            if text_lens > codec_lens:
                return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
            else:
                # 情况 B：文字比编号短 -> 把文字末尾用 tts_pad 补到 codec_lens 长，整段拼起来；
                # 没有剩余文字，trailing 用 tts_pad_embed 占位
                text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
                return text_embed + codec_embed, tts_pad_embed

    # 主生成入口：把文字 + (可选)instruct + (可选)参考音频/说话人向量拼成 talker 输入，
    # 然后调 talker.generate 自回归生成 16 层语音编号。
    #
    # 输入关键参数：
    #   input_ids          : 每条样本的文字编号列表（含 <|im_start|>assistant 等控制标记）
    #   instruct_ids       : 可选 instruct 文本编号（如情感/风格指令）
    #   ref_ids            : ICL 模式下，参考音频对应的文字编号
    #   voice_clone_prompt : 声音克隆相关（x-vector、ICL 参考编号、模式开关等）
    #   languages/speakers : 每条样本的语言与说话人
    #   non_streaming_mode : 是否非流式（文字整段在前，不与编号交错）
    #   采样参数            : top_k/top_p/temperature 控制 talker，subtalker_* 控制子 talker
    #
    # 输出：(talker_codes_list, talker_hidden_states_list)
    #   - talker_codes_list        : 每条样本实际生成的 16 层编号（已按 eos 截断）
    #   - talker_hidden_states_list: 每条样本对应的 talker hidden（用于下游 code_predictor 训练或调试）
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: list[dict] = None,
        languages: list[str] = None,
        speakers: list[str] = None,
        non_streaming_mode = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ):
        # talker.generate 的全部超参：含主采样参数 + 子 talker 采样参数 + eos + 抑制 token 列表。
        # suppress_tokens 把词表后 1024 个特殊 token（除 codec_eos）都禁掉，避免生成无意义标记。
        talker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 2,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "subtalker_dosample": subtalker_dosample,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "eos_token_id": eos_token_id
            if eos_token_id is not None
            else self.config.talker_config.codec_eos_token_id,
            "repetition_penalty": repetition_penalty,
            "suppress_tokens": [
                i
                for i in range(self.config.talker_config.vocab_size - 1024, self.config.talker_config.vocab_size)
                if i not in (self.config.talker_config.codec_eos_token_id,)
            ],
            "output_hidden_states": getattr(kwargs, "output_hidden_states", True),
            "return_dict_in_generate": getattr(kwargs, "return_dict_in_generate", True)
        }

        # talker_input_embeds[index] 是一个 list，先把 instruct（如果有）和主提示拼进去
        talker_input_embeds = [[] for _ in range(len(input_ids))]

        voice_clone_spk_embeds = None
        # voice clone speaker prompt generate
        # 声音克隆：把外部提好的 x-vector 整理成 talker 可用的说话人向量列表
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)

        # instruct text prompt generate
        # 指令文本（可选）：直接过 text_projection 拼到该样本提示最前面
        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(self.talker.text_projection(
                                                  self.talker.get_text_embeddings()(instruct_id)))

        # tts text prompt generate
        # 主循环：对每条样本拼出 talker 输入嵌入
        trailing_text_hiddens = []
        if speakers is None:
            speakers = [None] * len(input_ids)
        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):
            # 1) 决定 speaker_embed：
            #    - 没有声音克隆：按名字查预置 spk_id 取 embedding；空名字 -> None（走 instruct）
            #    - 有声音克隆：x_vector_only_mode 或 icl_mode 时用参考向量，否则 None
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker == None: # Instruct create speaker
                    speaker_embed = None
                else:
                    if speaker.lower() not in self.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    else:
                        spk_id = self.config.talker_config.spk_id[speaker.lower()]
                        speaker_embed = self.talker.get_input_embeddings()(
                                            torch.tensor(
                                                spk_id,
                                                device=self.talker.device,
                                                dtype=input_id.dtype,
                                            )
                                        )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            assert language is not None

            # 2) 决定 language_id：auto 时不指定，否则按 codec_language_id 取；
            #    若该 speaker 是方言且语言是中文/auto，则改用方言对应的 language_id
            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in self.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                else:
                    language_id = self.config.talker_config.codec_language_id[language.lower()]

            if (language.lower() in ["chinese", "auto"] and \
                   speaker != "" and speaker is not None and \
                     self.config.talker_config.spk_is_dialect[speaker.lower()] != False):
                dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = self.config.talker_config.codec_language_id[dialect]

            # 3) tts 控制（bos/eos/pad）的 embedding，整段都共用
            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(
                    torch.tensor(
                        [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                        device=self.talker.device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)  # 3 * [1 1 d]

            # 4) codec 控制标记（think/nothink、bos/eos、language_id 等）的 embedding 段。
            #    根据是否有 language_id 选不同前缀；speaker_embed（如果有）插在两段中间。
            # codec: tag and speaker
            if language_id is None:
                codec_prefill_list = [[
                                        self.config.talker_config.codec_nothink_id,
                                        self.config.talker_config.codec_think_bos_id,
                                        self.config.talker_config.codec_think_eos_id,
                                    ]]
            else:
                codec_prefill_list = [[
                                        self.config.talker_config.codec_think_id,
                                        self.config.talker_config.codec_think_bos_id,
                                        language_id,
                                        self.config.talker_config.codec_think_eos_id,
                                    ]]

            codec_input_emebdding_0 = self.talker.get_input_embeddings()(
                                                    torch.tensor(
                                                        codec_prefill_list,
                                                        device=self.talker.device,
                                                        dtype=input_id.dtype,
                                                    )
                                                )
            codec_input_emebdding_1 = self.talker.get_input_embeddings()(
                                                    torch.tensor(
                                                        [[
                                                            self.config.talker_config.codec_pad_id,
                                                            self.config.talker_config.codec_bos_id,
                                                        ]],
                                                        device=self.talker.device,
                                                        dtype=input_id.dtype,
                                                    )
                                                )
            if speaker_embed is None:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0,
                                                   codec_input_emebdding_1], dim=1)
            else:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0,
                                                   speaker_embed.view(1, 1, -1),
                                                   codec_input_emebdding_1], dim=1)

            # '<|im_start|>assistant\n我叫通义千问，是阿里云的开源大模型。<|im_end|>\n<|im_start|>assistant\n'

            # 5) 角色标记 <|im_start|>assistant\n 的 embedding（前 3 个 token）
            # <|im_start|>assistant\n
            _talker_input_embed_role = self.talker.text_projection(
                                        self.talker.get_text_embeddings()(input_id[:, :3])
                                        )

            # 6) 把 codec 控制段（[:-1] 即去掉最后一个 codec_bos）与"tts_pad * N + tts_bos"对齐相加。
            #    含义：codec 控制段每个位置都"叠加"一个 tts 控制（pad 或 bos），
            #    这样 talker 在每个 codec 控制位置也看到对应 tts 的角色/位置信息。
            # tts_pad * 4 + tts_bos
            _talker_input_embed = torch.cat((tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
                                            tts_bos_embed,
                                            ), dim=1) + codec_input_emebdding[:, :-1]

            talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

            # 7) ICL 模式：把参考音频编号 + 参考文字 + 待合成文字拼成 ICL 提示并接在后面。
            #    非 ICL 模式：只把文字第一个 token 接上，后面按流式/非流式分别构造 trailing_text_hidden。
            if voice_clone_prompt is not None and voice_clone_prompt["ref_code"] is not None and voice_clone_prompt["icl_mode"][index]:
                icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(self.talker.device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                #  tts_text_first_token
                # 非流式/流式公共部分：把待合成文字的第一个 token（与最后一个 codec 控制对齐）接上
                talker_input_embed = torch.cat([talker_input_embed,
                                                self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_emebdding[:, -1:]],
                                                dim=1)
                if non_streaming_mode:
                    # 非流式：把整段待合成文字 + eos 一次性拼进来（每步对齐 codec_pad），
                    # 再接一个"tts_pad + codec_bos"作为语音段起点。
                    talker_input_embed = talker_input_embed[:, :-1] # 去掉原本放进去的text
                    talker_input_embed = torch.cat([talker_input_embed,
                                                    torch.cat((self.talker.text_projection(
                                                        self.talker.get_text_embeddings()(input_id[:, 3:-5])
                                                    ), tts_eos_embed), dim=1) + self.talker.get_input_embeddings()(
                                                        torch.tensor(
                                                            [[
                                                                self.config.talker_config.codec_pad_id,
                                                            ] * (input_id[:, 3:-5].shape[1] + 1)],
                                                            device=self.talker.device,
                                                            dtype=input_id.dtype,
                                                        )
                                                    ),
                                                    tts_pad_embed + self.talker.get_input_embeddings()(
                                                        torch.tensor(
                                                            [[
                                                                self.config.talker_config.codec_bos_id,
                                                            ]],
                                                            device=self.talker.device,
                                                            dtype=input_id.dtype,
                                                        )
                                                    )
                                                    ], dim=1)
                    # 非流式没有 trailing，统一用 pad
                    trailing_text_hidden = tts_pad_embed
                else:
                    # 流式：trailing_text_hidden = 待合成文字（去掉首 token） + eos
                    # 之后在 talker 自回归每步会按 generation_step 取一位加到输入里，
                    # 实现"文字一边读一边出语音编号"的流式对齐
                    # 叫通义千问，是阿里云的开源大模型。
                    trailing_text_hidden = torch.cat((self.talker.text_projection(
                                                        self.talker.get_text_embeddings()(input_id[:, 4:-5])
                                                    ), tts_eos_embed), dim=1)
            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        # 把每条样本的 instruct 段 + 主提示段沿时间维拼成一条
        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

        # 8) 左 padding：把不同长度的样本对齐到同一长度，方便 batch 推理。
        #    做法：先翻转序列 -> pad_sequence 填 0 -> 再翻回来，等价于在序列左侧补 0。
        # for batch inferquence
        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        # left padding for talker input embeds
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed,
            batch_first=True,
            padding_value=0.0
        )
        talker_input_embeds = padded_reversed.flip(dims=[1])
        # generate mask
        # 构造 attention_mask：有效位置 1，左 pad 位置 0
        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)
        # padding trailing text hiddens
        # trailing_text_hidden 也要 pad 到同一长度，且超出的位置用 tts_pad_embed 填充（不是 0），
        # 因为之后会把它直接加到 talker 输入上。
        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(
            sequences_to_pad,
            batch_first=True,
            padding_value=0.0
        )
        arange_tensor = torch.arange(max(trailing_text_original_lengths),
                                     device=padded_hiddens.device).expand(len(trailing_text_original_lengths), -1)
        lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens

        # 9) 真正调 talker.generate 自回归生成语音编号
        # forward
        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            **talker_kwargs,
        )

        # 10) 整理输出：
        #   - talker_codes       : 把每步 hidden_states[-1]（即 codec_ids 16 层编号）stack 起来
        #   - talker_hidden_states: 取每步最后一层 hidden，拼起来供后续训练/调试
        talker_codes = torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)
        talker_hidden_states = torch.cat([hid[0][-1][:, -1:] for hid in talker_result.hidden_states], dim=1)[:, :-1]

        # 11) 按 codec_eos_token_id 截断：第 0 层编号里出现 eos 即认为该样本已结束。
        first_codebook = talker_codes[:, :, 0]
        is_stop_token = (first_codebook ==  self.config.talker_config.codec_eos_token_id)
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])

        # 12) 按每条样本的有效长度切片，去掉尾部的 pad/eos
        talker_codes_list = [talker_codes[i, :length, ] for i, length in enumerate(effective_lengths)]
        talker_hidden_states_list = [talker_hidden_states[i, :length, :] for i, length in enumerate(effective_lengths)]
        
        return talker_codes_list, talker_hidden_states_list

__all__ = [
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSTalkerModel",
]
