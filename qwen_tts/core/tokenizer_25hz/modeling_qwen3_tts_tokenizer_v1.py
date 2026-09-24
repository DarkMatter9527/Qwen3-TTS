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
"""PyTorch Qwen3TTSTokenizerV1 model."""
# ====================================================================================
# 文件总览：Qwen3-TTS 早期版（25Hz）语音 tokenizer 模型实现
# ====================================================================================
# 职责：
#   把"音频 ↔ 离散编号"两种表示互相转换，作为 TTS 系统中的语音分词器。
#
# encode 流程（音频 → 编号）：
#   1) waveform → mel 频谱图（get_mel_audio，声音的"时间×频率"二维表示，
#      频率轴按人耳 mel 刻度拉伸，见 vq/whisper_encoder.py）
#   2) mel → Whisper 风格 Transformer 编码（WhisperEncoderVQ，见 vq/speech_vq.py）
#   3) 编码特征 → GRVQ 分组残差向量量化，从 32768 项码本中每步选 1 个编号
#   4) 每秒 25 步，每步 1 个编号（与 12Hz 版每秒 12 步不同）
#   5) 同时用 ECAPA-TDNN / XVectorExtractor 抽取说话人向量 x-vector 和参考 mel
#
# decode 流程（编号 → 波形）：
#   1) 编号 → embedding（DiTCodecEmbedding）
#   2) DiT 扩散 Transformer（先加噪再去噪）从编号 + 说话人向量 + 参考 mel 生成 mel 频谱
#   3) BigVGAN 声码器把 mel 频谱变回可听波形
#
# 与 12Hz 版的关键区别：
#   - 帧率 25Hz（12Hz 版为 12Hz）
#   - decode 必须同时输入 audio_codes + xvectors(说话人向量) + ref_mels(参考mel) 三项；
#     12Hz 版只需 codes 一项即可解码。
#   - 25Hz 版为早期未对外发布版本。
#
# 架构位置：Qwen3TTSTokenizerV1Model 是顶层，含 encoder 和 decoder 两部分。
#   - encoder = Qwen3TTSTokenizerV1Encoder（含 WhisperEncoderVQ）
#   - decoder = Qwen3TTSTokenizerV1Decoder（含 DiT + BigVGAN）
#   - 另含 encoder_xvector_extractor（ECAPA-TDNN 风格说话人编码器，ONNX 模型）
# ====================================================================================

import math
from dataclasses import dataclass
from typing import Optional, Union, List

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import ModelOutput, auto_docstring, logging
from transformers.utils.hub import cached_file

from torch.nn.utils.rnn import pad_sequence

# whisper_encoder 提供 waveform → mel 与 CNN 后序列长度计算工具
from .vq.whisper_encoder import get_mel_audio, get_T_after_cnn
# speech_vq 提供 Whisper 风格 Transformer + GRVQ 量化器，以及说话人 x-vector 抽取器
from .vq.speech_vq import WhisperEncoderVQ, XVectorExtractor

from .configuration_qwen3_tts_tokenizer_v1 import (
    Qwen3TTSTokenizerV1Config,
    Qwen3TTSTokenizerV1EncoderConfig,
    Qwen3TTSTokenizerV1DecoderConfig,
    Qwen3TTSTokenizerV1DecoderBigVGANConfig,
    Qwen3TTSTokenizerV1DecoderDiTConfig
)

logger = logging.get_logger(__name__)


# ====================================================================================
# 数据结构 & 基类定义
# ====================================================================================

@dataclass
@auto_docstring
class Qwen3TTSTokenizerV1EncoderOutput(ModelOutput):
    """编码器输出：编号 + 说话人向量 + 参考 mel 三件套（25Hz 版必需）。"""

    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discret code embeddings computed using `model.encode`, each tensor has shape (codes_length_i,).
    xvectors (`List[torch.FloatTensor]`):
        X-vector embeddings computed using `model.encode`, each tensor has shape (xvector_dim,).
    ref_mels (`List[torch.FloatTensor]`):
        Reference mel spectrogram computed using `model.encode`, each tensor has shape (mel_length_i, mel_dim,).
    """

    # 离散编号序列：每秒 25 步，每步从 32768 项码本选 1 个编号
    audio_codes: List[torch.LongTensor] = None
    # 说话人身份向量 x-vector（用于 decode 时条件控制音色）
    xvectors: List[torch.FloatTensor] = None
    # 参考音频的 mel 频谱（decode 时作为条件输入）
    ref_mels: List[torch.FloatTensor] = None


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV1DecoderOutput(ModelOutput):
    """解码器输出：最终生成的波形列表。"""

    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio values, obtained using the decoder part of Qwen3TTSTokenizerV1.
        Each tensor has shape (segment_length_i).
    """

    # 解码后的音频波形，每个样本一段一维 float 张量
    audio_values: List[torch.FloatTensor] = None


@auto_docstring
class Qwen3TTSTokenizerV1DecoderPreTrainedModel(PreTrainedModel):
    """解码器（DiT + BigVGAN）的预训练基类，统一管理权重加载、注意力后端等。"""
    config: Qwen3TTSTokenizerV1DecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"  # KV cache 不参与设备放置优化
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False  # 含动态结构，无法整图编译
    _supports_attention_backend = True


@auto_docstring
class Qwen3TTSTokenizerV1EncoderPreTrainedModel(PreTrainedModel):
    """编码器（Whisper + GRVQ）的预训练基类。"""
    config: Qwen3TTSTokenizerV1EncoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


# ====================================================================================
# DiT 旋转位置编码（RoPE）
# 通过把 Q/K 在不同维度上做不同频率的旋转，把相对位置信息注入注意力，无需可学习的位置嵌入。
# ====================================================================================
class Qwen3TTSTokenizerV1DecoderDiTRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim, base=10000):
        super().__init__()
        # 不同维度的频率倒数：低维高频，高维低频，形成多尺度位置编码
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x 仅用来拿 batch/seq_len 和 dtype，本身不参与计算
        batch_size, seq_len = x.shape[0], x.shape[1]
        t = torch.arange(seq_len, device=x.device)
        device_type = x.device.type
        # mps 不支持部分浮点操作，回退到 cpu 计算
        device_type = device_type if device_type != "mps" else "cpu"
        # 关闭 autocast，保证 RoPE 始终在 fp32 下计算，避免精度问题
        with torch.autocast(device_type=device_type, enabled=False):
            # 位置 t × 频率倒数 → 每个 head_dim 维度的旋转角度
            freqs = t.unsqueeze(1).float() @ self.inv_freq.unsqueeze(0).float()
            # 把每对相邻角度堆叠成 cos/sin 的输入形式
            freqs = torch.stack((freqs, freqs), dim=-1)
            freqs = freqs.reshape(*freqs.shape[:-2], -1)
            # 沿 batch 维度复制（不同样本共用相同位置编码）
            freqs = freqs.repeat(batch_size, *([1] * freqs.dim()))
            cos = freqs.cos()
            sin = freqs.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ====================================================================================
# ECAPA-TDNN 风格说话人编码器（用于从参考 mel 提取 x-vector）
# 这套模块是经典说话人识别网络，用于 25Hz 版 decode 的说话人条件输入。
# 由 TDNN + Res2Net + Squeeze-Excitation + AttentiveStatisticsPooling 组成。
# ====================================================================================

class TimeDelayNetBlock(nn.Module):
    """TDNN 时延神经网络基本块：1D 卷积 + ReLU。
    padding='same' + reflect 模式让输出长度不变。"""

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
            padding_mode="reflect",  # 边界反射填充，避免零填充引入虚假信号
        )
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor):
        return self.activation(self.conv(hidden_states))


class Res2NetBlock(torch.nn.Module):
    """Res2Net 块：把通道分 scale 段，段间残差相连。
    在不增加参数量的前提下扩大感受野，是 ECAPA-TDNN 的核心。"""

    def __init__(self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1):
        super().__init__()

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        # scale-1 个 TDNN 子块，第 0 段直接透传
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
        outputs = []
        # 沿通道维度切成 scale 段，逐段处理并叠加残差
        for i, hidden_part in enumerate(torch.chunk(hidden_states, self.scale, dim=1)):
            if i == 0:
                output_part = hidden_part            # 第 0 段直接透传
            elif i == 1:
                output_part = self.blocks[i - 1](hidden_part)
            else:
                # 后续段都把"当前输入 + 上一段输出"送入对应 TDNN，形成层级残差
                output_part = self.blocks[i - 1](hidden_part + output_part)
            outputs.append(output_part)
        output = torch.cat(outputs, dim=1)
        return output


class SqueezeExcitationBlock(nn.Module):
    """SE 通道注意力：用全局均值描述通道重要性，再用 sigmoid 加权原特征。
    让网络学会"放大有用通道、抑制无用通道"。"""

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
        # 时间维取均值，得到每通道的全局描述 [B, C, 1]
        hidden_states_mean = hidden_states.mean(dim=2, keepdim=True)

        # 两层 1x1 卷积算通道权重
        hidden_states_mean = self.relu(self.conv1(hidden_states_mean))
        hidden_states_mean = self.sigmoid(self.conv2(hidden_states_mean))

        # 用通道权重加权原特征
        return hidden_states * hidden_states_mean


class AttentiveStatisticsPooling(nn.Module):
    """注意力统计池化：用注意力权重把变长序列压成定长说话人向量。
    返回 [均值 || 标准差] 拼接，让下游同时获得一阶和二阶统计量。
    原始实现见 ECAPA-TDNN 论文（HuggingFace papers/2005.07143）。"""

    def __init__(self, channels, attention_channels=128):
        super().__init__()

        self.eps = 1e-12  # 防止开方差为零
        # 输入是 [hidden || mean || std] 三段拼接，所以 in=channels*3
        self.tdnn = TimeDelayNetBlock(channels * 3, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        # 输出每帧每通道的注意力分数
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
        # 比位置 < 长度的位置标 True，构造 valid mask
        mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len
        ) < length.unsqueeze(1)

        mask = torch.as_tensor(mask, dtype=dtype, device=device)
        return mask

    def _compute_statistics(self, x, m, dim=2):
        """按权重 m 计算加权均值与加权标准差。"""
        mean = (m * x).sum(dim)
        std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(self.eps))
        return mean, std

    def forward(self, hidden_states):
        # 这里所有帧都视为有效（lengths 全 1 * seq_length），故 mask 全 True
        seq_length = hidden_states.shape[-1]
        lengths = torch.ones(hidden_states.shape[0], device=hidden_states.device)

        # Make binary mask of shape [N, 1, L]
        mask = self._length_to_mask(
            lengths * seq_length, max_len=seq_length, dtype=hidden_states.dtype, device=hidden_states.device
        )
        mask = mask.unsqueeze(1)

        # 先用均匀权重算一版 mean/std，作为注意力的额外上下文
        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        total = mask.sum(dim=2, keepdim=True)

        mean, std = self._compute_statistics(hidden_states, mask / total)
        # 把 mean/std 复制到每个时刻，与原特征拼接
        mean = mean.unsqueeze(2).repeat(1, 1, seq_length)
        std = std.unsqueeze(2).repeat(1, 1, seq_length)
        attention = torch.cat([hidden_states, mean, std], dim=1)

        # Apply layers
        # 通过 TDNN+Tanh+Conv 算每个时刻的注意力分数
        attention = self.conv(self.tanh(self.tdnn(attention)))

        # Filter out zero-paddings
        attention = attention.masked_fill(mask == 0, float("-inf"))

        # 时间维 softmax → 得到归一化注意力权重
        attention = F.softmax(attention, dim=2)
        # 用注意力权重重新统计 mean/std，作为最终说话人向量
        mean, std = self._compute_statistics(hidden_states, attention)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SqueezeExcitationRes2NetBlock(nn.Module):
    """ECAPA-TDNN 的核心残差块：TDNN → Res2Net → TDNN → SE，外加残差连接。"""

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
        # 入口/出口都是 1x1 卷积做通道变换
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
        # 残差：输入直接加到输出上
        residual = hidden_state

        hidden_state = self.tdnn1(hidden_state)
        hidden_state = self.res2net_block(hidden_state)
        hidden_state = self.tdnn2(hidden_state)
        hidden_state = self.se_block(hidden_state)

        return hidden_state + residual


class ECAPA_TimeDelayNet(torch.nn.Module):
    """ECAPA-TDNN 说话人编码器主体。
    从参考 mel 抽取定长说话人向量（x-vector），作为 25Hz 版 decode 的音色条件。
    论文："ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
    in TDNN Based Speaker Verification" (https://huggingface.co/papers/2005.07143)。"""

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderBigVGANConfig):
        super().__init__()
        # 三组通道/卷积核/膨胀配置长度必须一致
        if len(config.enc_channels) != len(config.enc_kernel_sizes) or len(config.enc_channels) != len(
            config.enc_dilations
        ):
            raise ValueError("enc_channels, enc_kernel_sizes and enc_dilations should have same length")
        self.channels = config.enc_channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        # 第一层 TDNN：把 mel 维度映射到 enc_channels[0]
        self.blocks.append(
            TimeDelayNetBlock(
                config.mel_dim,
                config.enc_channels[0],
                config.enc_kernel_sizes[0],
                config.enc_dilations[0],
            )
        )

        # SE-Res2Net layers
        # 中间若干层都是 SE-Res2Net 残差块
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
        # MFA：把中间各层输出拼接后再做一次 TDNN
        self.mfa = TimeDelayNetBlock(
            config.enc_channels[-1],
            config.enc_channels[-1],
            config.enc_kernel_sizes[-1],
            config.enc_dilations[-1],
        )

        # Attentive Statistical Pooling
        # ASP：把变长序列压成定长向量
        self.asp = AttentiveStatisticsPooling(
            config.enc_channels[-1],
            attention_channels=config.enc_attention_channels,
        )

        # Final linear transformation
        # 最后 1x1 卷积把维度映射到 enc_dim（x-vector 维度）
        # *2 是因为 ASP 输出 mean 和 std 拼接
        self.fc = nn.Conv1d(
            in_channels=config.enc_channels[-1] * 2,
            out_channels=config.enc_dim,
            kernel_size=1,
            padding="same",
            padding_mode="reflect",
        )

    def forward(self, hidden_states):
        # 输入 [B, T, mel_dim]，转成 [B, mel_dim, T] 给 Conv1d
        # Minimize transpose for efficiency
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states_list = []
        # 依次跑每一层，并记录每层输出用于 MFA
        for layer in self.blocks:
            hidden_states = layer(hidden_states)
            hidden_states_list.append(hidden_states)

        # Multi-layer feature aggregation
        # 跳过第 0 层（初始 TDNN），拼接中间各 SE-Res2Net 块的输出
        hidden_states = torch.cat(hidden_states_list[1:], dim=1)
        hidden_states = self.mfa(hidden_states)

        # Attentive Statistical Pooling
        hidden_states = self.asp(hidden_states)

        # Final linear transformation
        hidden_states = self.fc(hidden_states)

        # [B, enc_dim, 1] → [B, enc_dim]
        hidden_states = hidden_states.squeeze(-1)
        return hidden_states


# ====================================================================================
# DiT 解码器：从编号 + 说话人向量 + 参考 mel 通过扩散生成 mel 频谱
# 下面一组模块是 DiT（Diffusion Transformer）的标准组件：
#   - DiTInputEmbedding：把 [噪声mel | 参考 mel 编码 | code embedding | 说话人向量] 拼成 Transformer 输入
#   - DiTCodecEmbedding：把离散编号 → embedding 并按 repeats 上采样（25 步/秒 → mel 帧率）
#   - AdaLayerNormZero / _Final：用 timestep embedding 调制的 LayerNorm，实现条件化归一化
#   - DiTMLP：标准 FFN
#   - apply_rotary_pos_emb / DiTAttention：RoPE + 多头注意力
# ====================================================================================

class DiTInputEmbedding(nn.Module):
    """DiT 输入嵌入层：把四种条件信号拼接后投影到 hidden_size。
    - hidden_states：扩散过程中加噪的 mel（要去噪的目标）
    - speaker_embedding：x-vector 说话人向量（已编码后）
    - condition_vector：参考 mel（送入 spk_encoder 再编码一次）
    - code_embed：编号 → embedding
    支持 CFG（classifier-free guidance）：把条件与无条件样本沿 batch 拼接送入网络，省一半前向。"""

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderBigVGANConfig):
        super().__init__()
        # 输入维度 = mel_dim(噪声mel) + enc_dim(spk_encoder输出) + enc_emb_dim(code_emb) + emb_dim(说话人向量)
        # 输出投影到 DiT 的 hidden_size
        self.proj = nn.Linear(
            config.mel_dim + config.enc_dim + config.enc_emb_dim + config.emb_dim,
            config.hidden_size,
        )
        # 用 ECAPA-TDNN 对参考 mel 再次编码，得到时间维度的说话人条件特征
        self.spk_encoder = ECAPA_TimeDelayNet(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        speaker_embedding: torch.Tensor,
        condition_vector: torch.Tensor,
        code_embed: torch.Tensor,
        drop_audio_cond: Optional[bool] = False,
        code_embed_uncond: Optional[bool] = None,
        apply_cfg: Optional[bool] = True,
    ):
        if apply_cfg:
            # CFG 模式：把条件样本和无条件样本沿 batch 维拼接，整体只跑一次前向
            # 无条件样本：说话人向量、参考 mel 全置零；编号用"无条件编号"
            hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
            speaker_embedding = torch.cat([speaker_embedding, torch.zeros_like(speaker_embedding)], dim=0)
            condition_vector = torch.cat([condition_vector, torch.zeros_like(condition_vector)], dim=0)
            code_embed = torch.cat([code_embed, code_embed_uncond], dim=0)
        elif drop_audio_cond:  # cfg for cond audio
            # 训练时随机丢弃音色条件（不丢弃编号），提高鲁棒性
            condition_vector = torch.zeros_like(condition_vector)
            speaker_embedding = torch.zeros_like(speaker_embedding)
        # 参考 mel → 编码特征，再沿时间维复制到与 hidden_states 等长
        condition_vector = self.spk_encoder(condition_vector).unsqueeze(1).repeat(1, hidden_states.size(1), 1)
        # 四路条件拼接后线性投影到 hidden_size
        hidden_states = self.proj(torch.cat((hidden_states, condition_vector, code_embed, speaker_embedding), dim=-1))

        return hidden_states


# Transformer backbone using DiT blocks
class DiTCodecEmbedding(nn.Module):
    """把离散编号 embedding 后按 repeats 在时间维复制，把 25 步/秒的编号上采样到 mel 帧率。"""

    def __init__(self, codec_num_embeds, codec_dim, repeats):
        super().__init__()
        self.repeats = repeats
        # +1 是为了容纳"无条件"编号（CFG 用 0 作为无条件编号）
        self.codec_embed = nn.Embedding(codec_num_embeds + 1, codec_dim)

    def forward(self, code, drop_code=False):
        # 训练时按一定概率把 code 全置零（变成无条件编号），用于 CFG 训练
        if drop_code:
            code = torch.zeros_like(code)
        code_embed = self.codec_embed(code)

        # 沿时间维每个编号重复 repeats 次，对齐 mel 帧率
        code_embed = torch.repeat_interleave(code_embed, repeats=self.repeats, dim=1)
        return code_embed


# AdaLayerNormZero
# return with modulated x for attn input, and params for later mlp modulation
class AdaLayerNormZero(nn.Module):
    """自适应 LayerNorm（DiT 风格）：用 timestep embedding 调制 scale/shift/gate，
    共输出 6 组参数（attn 与 mlp 各 scale/shift/gate），实现 timestep 条件化。"""

    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        # 一次产生 6 组调制参数：attn(s_shift, s_scale, s_gate) + mlp(s_shift, s_scale, s_gate)
        self.linear = nn.Linear(dim, dim * 6)

        # 不带可学习参数，完全由外部 modulation 控制
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb=None):
        # emb 是 timestep embedding
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)

        # 先做 LayerNorm，再用 scale_msa / shift_msa 调制（用于注意力前的输入）
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        # 返回调制后的 x，以及 MLP 用的 gate/shift/scale 供后续块使用
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


# AdaLayerNormZero for final layer
# return only with modulated x for attn input, cuz no more mlp modulation
class AdaLayerNormZero_Final(nn.Module):
    """最后一层的自适应 LayerNorm：只产生 scale/shift 两组参数，
    因为最后一层后面没有 MLP 需要调制。"""

    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)

        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states


# FeedForward
class DiTMLP(nn.Module):
    """标准 FFN：Linear → GELU(tanh近似) → Dropout → Linear，扩张倍率默认 4。"""

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)

        self.ff = nn.ModuleList(
            [
                nn.Linear(dim, inner_dim),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim),
            ]
        )

    def forward(self, hidden_states):
        for layer in self.ff:
            hidden_states = layer(hidden_states)
        return hidden_states


# Modified from Llama with a different rotate function, will fixed in next release
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """对 Q/K 应用旋转位置编码（RoPE）。
    把每对相邻维度看作复数 (x1, x2)，做角度旋转：
        x' = x * cos + rotate_half(x) * sin
    rotate_half_codec 把 (x1, x2) 旋转 90 度变成 (-x2, x1)。

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

    def rotate_half_codec(x):
        # x = rearrange(x, "... (d r) -> ... d r", r=2)
        # 把最后一维按对拆分，做 90 度旋转：(x1, x2) → (-x2, x1)
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(*x.shape[:-2], -1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_codec(q) * sin)
    k_embed = (k * cos) + (rotate_half_codec(k) * sin)
    return q_embed, k_embed


class DiTAttention(nn.Module):
    """DiT 多头自注意力。
    支持 RoPE 位置编码和滑动窗口式 attention_mask（用于 look_ahead/look_backward）。
    注意：本实现非因果（is_causal=False），通过 attention_mask 控制可见范围。"""

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderBigVGANConfig):
        super().__init__()

        self.config = config
        self.dim = config.hidden_size
        self.heads = config.num_attention_heads
        self.inner_dim = config.head_dim * config.num_attention_heads
        self.dropout = config.dropout
        self.is_causal = False  # 非因果注意力，靠 mask 控制可见范围

        self.to_q = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_k = nn.Linear(config.hidden_size, self.inner_dim)
        self.to_v = nn.Linear(config.hidden_size, self.inner_dim)

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, config.hidden_size), nn.Dropout(config.dropout)])

    def forward(
        self,
        hidden_states,  # noised input x
        position_embeddings=None,  # rotary position embedding for x
        attention_mask=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # `sample` projections.
        # Q/K/V 投影
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # attention
        # 拆头：[B, T, inner] → [B, heads, T, head_dim]
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # apply rotary position embedding
        # Due to training process, only first head is applied with RoPE, will be fixed at next release
        # 给 Q/K 注入相对位置信息
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # 调用 transformers 注册的注意力实现（sdpa / flash / eager 任选）
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_weights, _ = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask=attention_mask,
            is_causal=False,
        )

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        # 合并多头，转回 [B, T, inner_dim]
        attention_weights = attention_weights.reshape(batch_size, -1, self.heads * head_dim)
        attention_weights = attention_weights.to(query.dtype)

        # linear proj
        # 输出投影 + dropout
        attention_output = self.to_out[0](attention_weights)
        attention_output = self.to_out[1](attention_output)

        return attention_output


# time step conditioning embedding
class SinusPositionEmbedding(nn.Module):
    """正弦位置编码：常用于 timestep 这类标量条件。
    输出 [sin(t·freq), cos(t·freq)]，让后续 MLP 能区分不同 timestep。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, hidden_states, scale=1000):
        # hidden_states 是 [B] 的 timestep
        device = hidden_states.device
        half_dim = self.dim // 2
        # 频率呈几何级数，覆盖多尺度
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        # [B,1] * [1, half_dim] → [B, half_dim] 各维度的相位
        emb = scale * hidden_states.unsqueeze(1) * emb.unsqueeze(0)
        # 拼接 sin 和 cos → [B, dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.type_as(hidden_states)


class DiTTimestepEmbedding(nn.Module):
    """把 timestep 标量映射到 dim 维向量，作为 DiT 各层 AdaLayerNorm 的条件。
    结构：SinusPositionEmbedding → Linear → SiLU → Linear。"""

    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.ModuleList([nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)])

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        for layer in self.time_mlp:
            time_hidden = layer(time_hidden)  # b d
        return time_hidden


class DiTDecoderLayer(nn.Module):
    """DiT 单层：AdaLayerNorm → Attention → 残差 → LayerNorm → MLP → 残差。
    look_ahead/look_backward 控制本层注意力窗口：
      - look_ahead_block=1 表示当前块可以看到"下一个"块（前瞻）
      - look_backward_block=1 表示当前块可以看到"上一个"块（后顾）
    通过 block_diff 的相对块编号差构造滑动窗口 mask。"""

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderBigVGANConfig, look_ahead_block=0, look_backward_block=0):
        super().__init__()
        # 注意力前的自适应 LayerNorm（输出 attn 调制参数 + mlp 调制参数）
        self.attn_norm = AdaLayerNormZero(config.hidden_size)

        self.attn = DiTAttention(config)
        self.look_ahead_block = look_ahead_block
        self.look_backward_block = look_backward_block
        # MLP 前的 LayerNorm（仅 scale/shift 来自 attn_norm 输出，不再单独学参数）
        self.ff_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = DiTMLP(dim=config.hidden_size, mult=config.ff_mult, dropout=config.dropout)

    def forward(
        self, hidden_states, timestep, position_embeddings=None, block_diff=None
    ):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        # 用 timestep 调制 hidden_states，返回注意力用的归一化输入 + MLP 的调制参数
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(hidden_states, emb=timestep)

        # attention
        # 构造滑动窗口 mask：只允许块编号差落在 [-look_backward, +look_ahead] 范围的位置相互看见
        attn_output = self.attn(
            hidden_states=norm,
            position_embeddings=position_embeddings,
            attention_mask=(block_diff >= -float(self.look_backward_block))
            & (block_diff <= float(self.look_ahead_block)),
        )

        # process attention output for input x
        # 残差 + gate_msa 控制注意力贡献大小
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # MLP 前再做一次（带调制的）归一化
        norm = self.ff_norm(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        # 残差 + gate_mlp 控制 MLP 贡献大小
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        return hidden_states


# ====================================================================================
# BigVGAN 声码器组件
# BigVGAN 把 mel 频谱变回波形：先逐层上采样到目标采样率，每层带若干 AMPBlock 残差块。
# 下面这一组模块都是 BigVGAN 的标准件：
#   - SnakeBeta：周期性激活函数，对音频这种周期信号更友好
#   - kaiser_sinc_filter1d / UpSample1d / DownSample1d：抗混叠的上/下采样
#   - TorchActivation1d：把激活函数包在"上采样→激活→下采样"里，避免混叠
#   - CausalConv1d：因果 1D 卷积（仅左侧填充），用于实时流式合成
#   - AMPBlock：BigVGAN 的残差块（AMP = Anti-aliased Multi-Periodicity）
# ====================================================================================

class SnakeBeta(nn.Module):
    """SnakeBeta 周期激活函数：x + (1/β)·sin²(α·x)。
    α 控制频率，β 控制幅度，都是可学习参数。
    对音频这种本质周期信号比 ReLU/GELU 更适合。

    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://huggingface.co/papers/2006.08195
    """

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        # 初始为 0，前向时会用 exp 放大成 1，保证训练初期接近恒等映射
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001  # 防止除零

    def forward(self, hidden_states):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        # 把 [C] 扩展成 [1, C, 1] 与输入 [B, C, T] 对齐
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        # exp 保证 α/β 恒为正
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )

        return hidden_states


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """生成 1D Kaiser 窗 sinc FIR 滤波器（抗混叠用）。
    通过 Kaiser 窗压低旁瓣，sinc 函数实现低通，用于上/下采样前后防止频率混叠。

    Args:
        cutoff (float): Normalized cutoff frequency (0 to 0.5).
        half_width (float): Transition bandwidth.
        kernel_size (int): Number of filter taps.

    Returns:
        torch.Tensor: A tensor of shape (1, 1, kernel_size) representing the filter.
    """
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Compute Kaiser window parameters
    # 由过渡带宽推目标阻带衰减，再推 Kaiser β
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)

    # Compute time indices
    # 偶数核时间轴偏 0.5，奇数核居中
    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size

    # Compute sinc filter
    if cutoff == 0:
        return torch.zeros((1, 1, kernel_size), dtype=torch.float32)  # Ensures correct shape

    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter

    # Normalize to ensure sum = 1 (avoid leakage of constant component)
    # 归一化使滤波器增益为 1，避免直流分量泄漏
    normalized_filter /= normalized_filter.sum()

    return normalized_filter.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    """1D 上采样（带抗混叠）：用转置卷积把时间维放大 ratio 倍。"""

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        # 默认核长 = 6*ratio//2*2，约为 ratio 的 6 倍且为偶数
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2

        # cutoff 取 0.5/ratio，正好对应上采样后奈奎斯特频率
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]

        # 左右复制填充，避免边界突变
        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        # 转置卷积做上采样，weight = sinc 滤波器，groups=channels 即每通道独立
        # 乘 ratio 保证幅度不下降
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        # 裁掉两侧因卷积产生的多余样本
        hidden_states = hidden_states[..., self.pad_left : -self.pad_right]

        return hidden_states


class DownSample1d(nn.Module):
    """1D 下采样（带抗混叠）：先用低通滤波再 stride 卷积缩小 ratio 倍。"""

    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio

        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        # 分组卷积 + stride=ratio 实现下采样
        out = F.conv1d(hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)
        return out


class TorchActivation1d(nn.Module):
    """把任意激活函数包成"上采样→激活→下采样"，相当于在高分辨率下做激活，
    再降回原分辨率，避免激活引入的高频分量被下采样混叠回来。"""

    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        if not callable(activation):
            raise TypeError("Activation function must be callable")
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, hidden_states):
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)

        return hidden_states


class CausalConv1d(nn.Conv1d):
    """因果 1D 卷积：所有 padding 都加在左侧（过去），
    保证时刻 t 的输出只依赖 t 及之前，用于流式生成。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 因果 padding = 膨胀 × (核-1)，全部加左侧
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        # 左侧 padding causal_padding，右侧 0
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class AMPBlock(torch.nn.Module):
    """BigVGAN 的残差块 AMP（Anti-aliased Multi-Periodicity）。
    内含 3 组 (Snake 激活 + 因果/非因果卷积 + Snake 激活 + 卷积)，每组的膨胀率不同，
    捕捉不同周期的语音结构。causal_type='2' 全部因果，用于流式；'1' 部分非因果。"""

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        causal_type='1',
    ):
        super().__init__()

        # convs1：三组膨胀卷积，dilation 递增，扩大感受野
        self.convs1 = nn.ModuleList(
            [
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                ),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                ),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                ),
            ]
        )

        if causal_type == '1':
            # type '1'：convs2 用普通非因果卷积（左右对称 padding）
            self.convs2 = nn.ModuleList(
                [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self._get_padding(kernel_size, 1),
                    ),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self._get_padding(kernel_size, 1),
                    ),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=self._get_padding(kernel_size, 1),
                    ),
                ]
            )
        else:
            # type '2'：convs2 也用因果卷积，全块因果
            self.convs2 = nn.ModuleList(
                [
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                    ),
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                    ),
                    CausalConv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                    ),
                ]
            )

        self.num_layers = len(self.convs1) + len(self.convs2)  # total number of conv layers

        # 每个卷积前都配一个抗混叠 Snake 激活
        self.activations = nn.ModuleList(
            [TorchActivation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

        if causal_type == '2':
            # type '2' 还多一个前置 conv+act 做预处理
            self.pre_conv = nn.Conv1d(
                                channels,
                                channels,
                                kernel_size,
                                stride=1,
                                padding=self._get_padding(kernel_size, 1),
                            )
            self.pre_act = TorchActivation1d(activation=SnakeBeta(channels))
        else:
            self.pre_conv = nn.Identity()
            self.pre_act = nn.Identity()

    def _get_padding(self, kernel_size, dilation=1):
        # 对称 padding 让普通卷积保持长度不变
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, x):
        # type '2' 有预处理，'1' 直接透传
        hidden_states = self.pre_conv(x)
        hidden_states = self.pre_act(hidden_states)
        # activations 按奇偶分两组，分别配 conv1 和 conv2 前的激活
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            # act → 膨胀卷积 → act → 普通卷积，最后残差
            hidden_states = act1(hidden_states)
            hidden_states = conv1(hidden_states)
            hidden_states = act2(hidden_states)
            hidden_states = conv2(hidden_states)
            x = x + hidden_states  # 残差累加
        return x


@auto_docstring
class Qwen3TTSTokenizerV1DecoderBigVGANModel(Qwen3TTSTokenizerV1DecoderPreTrainedModel):
    """BigVGAN 声码器：把 mel 频谱 → 波形。
    结构：conv_pre → 多层(上采样 + 多个 AMPBlock 求平均) → Snake 激活 → conv_post 输出波形。
    每上采样一层，通道数减半（2 的幂次），最终卷到 1 通道的波形。"""
    config: Qwen3TTSTokenizerV1DecoderBigVGANConfig

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderBigVGANConfig):
        super().__init__(config)
        self.num_residual_blocks = len(config.resblock_kernel_sizes)
        self.num_upsample_layers = len(config.upsample_rates)

        # 入口 5-tap 卷积，把 mel_dim 通道映射到 upsample_initial_channel
        self.conv_pre = nn.Conv1d(config.mel_dim, config.upsample_initial_channel, 5, 1, padding=2)

        # Removing extra ModuleList breaks official state dict
        # 上采样层：每层一个 ConvTranspose1d，stride=upsample_rates[i]，通道减半
        ups = [
            nn.ModuleList(
                [
                    nn.ConvTranspose1d(
                        config.upsample_initial_channel // (2**layer_idx),
                        config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                        kernel_size,
                        stride,
                        padding=(kernel_size - stride) // 2,
                    )
                ]
            )
            for layer_idx, (stride, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes))
        ]
        self.ups = nn.ModuleList(ups)

        # 每个上采样层后接 num_residual_blocks 个 AMPBlock（不同 kernel/dilation 捕捉不同周期）
        # 前 2 层用 causal_type='2'（更严格因果），后面用 '1'
        self.resblocks = nn.ModuleList(
            [
                AMPBlock(config.upsample_initial_channel // (2 ** (layer_idx + 1)), kernel_size, dilation, '1' if layer_idx > 1 else '2')
                for layer_idx in range(self.num_upsample_layers)
                for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ]
        )

        self.activation_post = TorchActivation1d(
            activation=SnakeBeta(config.upsample_initial_channel // (2**self.num_upsample_layers))
        )
        # 最后 7-tap 卷积把通道压到 1（单声道波形），不带 bias
        self.conv_post = nn.Conv1d(
            config.upsample_initial_channel // (2**self.num_upsample_layers), 1, 7, 1, padding=3, bias=False
        )

    def normalize_spectrogram(self, spectrogram, max_value, min_db):
        # 把分贝谱归一化到 [-max_value, max_value]，方便卷积处理
        return torch.clamp((2 * max_value) * ((spectrogram - min_db) / (-min_db)) - max_value, -max_value, max_value)

    def amplitude_to_db(self, amplitude, min_db_level):
        # 振幅 → 分贝：20·log10(amplitude)，min_level 防止 log(0)
        min_level = torch.exp(
            torch.tensor(min_db_level / 20.0 * np.log(10), device=amplitude.device, dtype=amplitude.dtype)
        )
        return 20 * torch.log10(torch.clamp(amplitude, min=min_level))

    def process_mel_spectrogram(self, mel_spectrogram):
        # DiT 输出的是 log-mel，先 exp 还原为线性振幅谱
        amplitude_spectrum = torch.exp(mel_spectrogram)
        # 转 dB 并减去 20 dB 偏置，再归一化
        decibel_spectrum = self.amplitude_to_db(amplitude_spectrum, -115) - 20
        return self.normalize_spectrogram(decibel_spectrum, 1, -115)

    def forward(self, mel_spectrogram):
        # 输入 mel 频谱 [B, mel_dim, T_mel]
        processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
        hidden_representation = self.conv_pre(processed_spectrogram)

        # 逐层上采样：每层先 ConvTranspose 把时间维放大，再过多个 AMPBlock 求平均
        for layer_index in range(self.num_upsample_layers):
            hidden_representation = self.ups[layer_index][0](hidden_representation)
            # 同层多个 AMPBlock 输出求和再平均
            residual_output = sum(
                self.resblocks[layer_index * self.num_residual_blocks + block_index](hidden_representation)
                for block_index in range(self.num_residual_blocks)
            )
            residual_output = residual_output / self.num_residual_blocks
            hidden_representation = residual_output

        # 末尾 Snake 激活 + 1x1 卷积输出单声道波形
        hidden_representation = self.activation_post(hidden_representation)
        output_waveform = self.conv_post(hidden_representation)
        # 限幅到 [-1, 1] 防止爆音
        return torch.clamp(output_waveform, min=-1.0, max=1.0).squeeze(1)


@auto_docstring
class Qwen3TTSTokenizerV1DecoderDiTModel(Qwen3TTSTokenizerV1DecoderPreTrainedModel):
    """DiT 解码器主体：从 (噪声mel + 说话人向量 + 参考 mel + 编号 + timestep) 预测速度场，
    配合采样器（sample 方法）做 flow matching / ODE 积分生成 mel 频谱。"""
    config: Qwen3TTSTokenizerV1DecoderDiTConfig
    _no_split_modules = ["DiTDecoderLayer"]

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderDiTConfig):
        super().__init__(config)
        self.mel_dim = config.mel_dim
        # repeats: 每个 25Hz 编号对应多少 mel 帧（决定上采样倍率）
        self.repeats = config.repeats
        # timestep → 条件向量，注入各层 AdaLayerNorm
        self.time_embed = DiTTimestepEmbedding(config.hidden_size)

        # 离散编号 → embedding 并按 repeats 上采样到 mel 帧率
        self.text_embed = DiTCodecEmbedding(config.num_embeds, config.emb_dim, config.repeats)
        # 把噪声mel + spk + ref_mel + code_embed 拼成 Transformer 输入
        self.input_embed = DiTInputEmbedding(config)

        # RoPE 旋转位置编码
        self.rotary_embed = Qwen3TTSTokenizerV1DecoderDiTRotaryEmbedding(config.head_dim)

        self.hidden_size = config.hidden_size
        self.layers = config.num_hidden_layers
        # block_size: 把 mel 序列按多大块分组用于 look_ahead/backward 滑窗注意力
        self.block_size = config.block_size
        self.num_attention_heads = config.num_attention_heads

        # 逐层构造 DiT block；look_ahead_layers / look_backward_layers 里的层
        # 会带 ±1 块的滑窗注意力，其他层默认只看本块
        self.transformer_blocks = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            self.transformer_blocks.append(
                DiTDecoderLayer(
                    config,
                    look_ahead_block=1 if i in config.look_ahead_layers else 0,
                    look_backward_block=1 if i in config.look_backward_layers else 0,
                )
            )

        self.norm_out = AdaLayerNormZero_Final(config.hidden_size)  # final modulation
        # 把 hidden_size 投回 mel_dim，得到预测的 mel 频谱
        self.proj_out = nn.Linear(config.hidden_size, config.mel_dim)

    def _create_block_diff(self, hidden_states):
        # 构造每对位置之间的"块编号差"矩阵 [B, heads, T, T]
        # 用于 look_ahead/backward 的滑动窗口 mask：只有块编号差落在窗口内才可见
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        block_indices = torch.arange(seq_len, device=hidden_states.device) // self.block_size  # [seq_length]

        block_i = block_indices.unsqueeze(1)  # [seq_length, 1]
        block_j = block_indices.unsqueeze(0)  # [1, seq_length]
        block_diff = block_j - block_i  # (n, n)

        return block_diff.expand(batch, self.num_attention_heads, seq_len, seq_len)

    def forward(
        self,
        hidden_states,
        condition_vector,
        speaker_embedding,
        quantized_code,
        time_step,
        drop_audio_conditioning=False,
        drop_code=False,
        apply_cfg=True,
    ):
        """训练时前向：给定 timestep t、加噪后的 mel x，预测 ODE 速度场 v(x,t)。
        apply_cfg=True 时把条件/无条件样本沿 batch 拼接，一次跑出两路输出供 CFG 用。"""
        # CFG 模式下 batch 翻倍（条件+无条件）
        batch_size = hidden_states.shape[0] * 2
        if time_step.ndim == 0:
            # 标量 timestep 复制到 batch 维
            time_step = time_step.repeat(batch_size)

        # Compute embeddings
        # timestep → 条件向量
        time_embedding = self.time_embed(time_step)
        # 编号 → embedding（CFG 时同时算条件/无条件两版）
        text_embedding = self.text_embed(quantized_code, drop_code=False if apply_cfg else drop_code)
        text_embedding_unconditioned = self.text_embed(quantized_code, drop_code=True) if apply_cfg else None

        # 把噪声 mel + 四路条件拼成 Transformer 输入 [B, T, hidden_size]
        hidden_states = self.input_embed(
            hidden_states,
            speaker_embedding,
            condition_vector,
            text_embedding,
            drop_audio_cond=drop_audio_conditioning,
            code_embed_uncond=text_embedding_unconditioned,
            apply_cfg=apply_cfg,
        )

        # Compute positional encodings
        # RoPE 位置编码 & 块差矩阵（用于滑窗 mask）
        position_embeddings = self.rotary_embed(hidden_states)
        blockwise_difference = self._create_block_diff(hidden_states)

        # Transformer blocks
        # 依次过每层 DiT block，hidden_states 形状不变
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(
                hidden_states,
                time_embedding,
                position_embeddings=position_embeddings,
                block_diff=blockwise_difference,
            )

        # 末层 AdaLN + 线性投影回 mel_dim
        hidden_states = self.norm_out(hidden_states, time_embedding)
        output = self.proj_out(hidden_states)

        return output

    def optimized_scale(self, positive_flat, negative_flat):
        """计算 CFG 缩放因子 st* = <v_cond, v_uncond> / ||v_uncond||²。
        用于动态调节 CFG 强度，比固定 guidance_scale 更稳。"""
        # Calculate dot production
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        # Squared norm of uncondition
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        # st_star = v_cond^T * v_uncond / ||v_uncond||^2
        st_star = dot_product / squared_norm
        return st_star

    @torch.no_grad()
    def sample(
        self,
        conditioning_vector,
        reference_mel_spectrogram,
        quantized_code,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        """采样生成 mel 频谱（flow matching / ODE 积分）。
        流程：
          1) 从纯高斯噪声开始（最大长度 30000 帧，按编号长度裁剪）
          2) 构造时间格点 linspace(0, 1, num_steps)，sway_coefficient 让步长偏置更靠后
          3) 每步用网络预测速度场 v(x, t)，按 Euler 法积分：x_{t+dt} = x_t + v·dt
          4) CFG：v = v_cond + (v_cond - v_uncond) * guidance_scale
        最终输出 mel 频谱 [B, mel_dim, T_mel]。"""
        # 30000 是预先分配的最大 mel 长度上界，再按实际编号长度裁剪
        noise_initialization = torch.randn([quantized_code.shape[0], 30000, self.mel_dim], dtype=reference_mel_spectrogram.dtype)
        maximum_duration = quantized_code.shape[1] * self.repeats
        initial_state = noise_initialization[:, :maximum_duration].to(quantized_code.device)
        # 说话人向量沿时间维复制，与 mel 等长
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, maximum_duration, 1)

        def ode_function(time_step, hidden_states):
            # guidance_scale 接近 0 时直接用条件预测，不做 CFG
            if guidance_scale < 1e-5:
                prediction = self(
                    hidden_states=hidden_states,
                    speaker_embedding=conditioning_vector,
                    condition_vector=reference_mel_spectrogram,
                    quantized_code=quantized_code,
                    time_step=time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                )
                return prediction

            # CFG：一次前向同时拿到条件/无条件两路输出
            model_output = self(
                hidden_states=hidden_states,
                quantized_code=quantized_code,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                time_step=time_step,
                apply_cfg=True,
            )
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)

            # CFG 公式：在 v_cond 方向推一点 (v_cond - v_uncond)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        initial_time = 0
        # 时间格点 linspace(0, 1, num_steps)
        time_embedding = torch.linspace(
            initial_time, 1, num_steps, device=quantized_code.device, dtype=conditioning_vector.dtype
        )

        # Sway sampling：对时间格点做非线性偏置，让步长在后期更细
        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        # Euler 积分：x_{t+dt} = x_t + v(x_t, t) * dt
        values = initial_state.clone()
        for t0, t1 in zip(time_embedding[:-1], time_embedding[1:]):
            dt = t1 - t0
            vt = ode_function(t0, values)
            values = values + vt * dt

        # [B, T, mel_dim] → [B, mel_dim, T]
        generated_mel_spectrogram = values.permute(0, 2, 1)
        return generated_mel_spectrogram


@auto_docstring
class Qwen3TTSTokenizerV1Decoder(Qwen3TTSTokenizerV1DecoderPreTrainedModel):
    """解码器顶层：组合 DiT（编号→mel）和 BigVGAN（mel→波形）两阶段。
    DiT 必须 fp32 推理，所以强制把注意力后端回退到 sdpa。"""
    config: Qwen3TTSTokenizerV1DecoderConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3TTSTokenizerV1DecoderDiTModel", "Qwen3TTSTokenizerV1DecoderBigVGANModel"]

    def __init__(self, config: Qwen3TTSTokenizerV1DecoderConfig):
        super().__init__(config)
        attn_impl = config._attn_implementation
        # DiT 必须 fp32 跑，flash_attention_2 只支持 fp16/bf16，故回退到 sdpa
        if config._attn_implementation == "flash_attention_2":
            logger.warning_once(
                "Qwen3TTSTokenizerV1Decoder must inference with fp32, but flash_attention_2 only supports fp16 and bf16, "
                "attention implementation of Qwen3TTSTokenizerV1Decoder will fallback to sdpa."
            )
            attn_impl = "sdpa"
        elif config._attn_implementation == "eager":
            # eager 注意力实现也不支持，回退 sdpa
            logger.warning_once(
                "Qwen3TTSTokenizerV1Decoder does not support eager attention implementation, fall back to sdpa"
            )
            attn_impl = "sdpa"
        # 子模型从各自 config 单独构造，统一使用选定的注意力后端
        self.dit = Qwen3TTSTokenizerV1DecoderDiTModel._from_config(
            config.dit_config, attn_implementation=attn_impl
        )
        self.bigvgan = Qwen3TTSTokenizerV1DecoderBigVGANModel._from_config(
            config.bigvgan_config, attn_implementation=attn_impl
        )

    def forward(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
        **kwargs,
    ):
        """两阶段生成波形：
        1) DiT 从 (编号 + xvector + ref_mel) 扩散采样出 mel 频谱
        2) BigVGAN 把 mel 频谱变成波形"""

        # 阶段一：DiT 扩散采样 → mel 频谱
        mel_spectrogram = self.dit.sample(
            conditioning,
            reference_mel,
            code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )

        # 阶段二：BigVGAN 声码器 → 波形
        waveform = self.bigvgan(mel_spectrogram)

        return waveform


class Qwen3TTSTokenizerV1Encoder(Qwen3TTSTokenizerV1EncoderPreTrainedModel):
    """编码器：whisper 风格 Transformer + GRVQ 量化，把音频变成离散编号。
    - speech2mel：waveform → mel 频谱
    - mel2code：mel 频谱 → 离散编号（每秒 25 步，从 32768 项码本选 1 个）
    - quantize_speech：上面两步的串联"""
    config: Qwen3TTSTokenizerV1EncoderConfig
    def __init__(self, config: Qwen3TTSTokenizerV1EncoderConfig):
        super().__init__(config)

        # WhisperEncoderVQ：Whisper 风格 Transformer + GRVQ 量化器
        # 详见 vq/speech_vq.py
        self.tokenizer = WhisperEncoderVQ(
            n_mels=config.n_mels,
            n_ctx=config.n_ctx,
            n_state=config.n_state,
            n_head=config.n_head,
            n_layer=config.n_layer,
            n_window=config.n_window,
            output_dim=config.output_dim,
            grad_checkpointing=config.grad_checkpointing,
            enable_mp=config.enable_mp,
            audio_sequence_parallel=config.audio_sequence_parallel,
            audio_vq_type=config.audio_vq_type,
            audio_vq_layers=config.audio_vq_layers,
            audio_vq_codebook_size=config.audio_vq_codebook_size,
            audio_vq_codebook_dim=config.audio_vq_codebook_dim,
            audio_vq_pe=config.audio_vq_pe,
            audio_vq_ds_rate=config.audio_vq_ds_rate,
        )

        self.padding = True
        # 编码器的下采样率（每秒 25 步就是由它决定的）
        self.audio_vq_ds_rate = self.tokenizer.audio_vq_ds_rate

    def speech2mel(self, speechs):
        # 对每条音频单独算 mel 频谱，padding 让长度对齐到 audio_vq_ds_rate 的倍数
        mels = [
            get_mel_audio(
                speech, padding = self.padding, audio_vq_ds_rate = self.audio_vq_ds_rate
            ).to(speech.dtype).to(self.tokenizer.conv1.weight.device)
            for speech in speechs
        ]
        return mels

    def mel2code(self, mels):
        # 计算 mel 长度、CNN 后长度、Transformer 序列长度（+2 是 Whisper 风格的边界 token）
        audio_mellens = [mel.size(-1) for mel in mels]
        audio_aftercnnlens = [get_T_after_cnn(T) for T in audio_mellens]
        audio_seqlens = [T + 2 for T in audio_aftercnnlens]

        with torch.no_grad():
            # 跑 WhisperEncoderVQ：返回量化特征 + 离散编号
            _, indices = self.tokenizer(
                x_list = mels,
                audio_mellens = audio_mellens,
                audio_aftercnnlens = audio_aftercnnlens,
                audio_seqlens = audio_seqlens,
                return_indices=True,
            )

        # 编号序列长度 = CNN 后长度 / 下采样率
        indice_lens = [T // self.tokenizer.audio_vq_ds_rate for T in audio_aftercnnlens]
        # 把变长编号 pad 成 batch 张量，padding_value=0
        indices  = pad_sequence(torch.split(indices, indice_lens), batch_first=True, padding_value=0)

        return indices, indice_lens

    def quantize_speech(self, speechs):
        # 串联 speech2mel + mel2code
        mels = self.speech2mel(speechs)
        indices, indice_lens = self.mel2code(mels)
        return indices, indice_lens


@auto_docstring
class Qwen3TTSTokenizerV1PreTrainedModel(PreTrainedModel):
    """顶层模型的预训练基类。"""
    config: Qwen3TTSTokenizerV1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizerV1 model.
    """
)
class Qwen3TTSTokenizerV1Model(Qwen3TTSTokenizerV1PreTrainedModel):
    """25Hz 版 tokenizer 顶层模型：encoder + decoder + x-vector 抽取器。
    外部调用 encode 把音频变 (codes, xvectors, ref_mels) 三件套；
    调用 decode 把这三件套变回波形。"""
    def __init__(self, config: Qwen3TTSTokenizerV1Config):
        super().__init__(config)
        self.config = config

        # 输入/输出采样率（输入波形采样率、解码后波形采样率）
        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate

        # 编号↔波形的时间步换算倍率（25Hz 编号 → 波形采样率的倍率）
        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        # 编码器：音频 → 离散编号
        self.encoder = Qwen3TTSTokenizerV1Encoder._from_config(self.config.encoder_config)
        # 解码器：编号 → 波形（含 DiT + BigVGAN）
        self.decoder = Qwen3TTSTokenizerV1Decoder._from_config(self.config.decoder_config)

        # 说话人向量抽取器（ONNX 形式的 campplus 模型），from_pretrained 时加载
        self.encoder_xvector_extractor = None

        self.post_init()

    def load_encoder_xvector_extractor(self, model_path):
        # 从 campplus.onnx 加载 ECAPA-TDNN 风格的说话人向量抽取器
        self.encoder_xvector_extractor = XVectorExtractor(model_path)

    def get_model_type(self):
        return self.config.model_type

    def get_input_sample_rate(self):
        return self.input_sample_rate

    def get_output_sample_rate(self):
        return self.output_sample_rate

    def get_encode_downsample_rate(self):
        return self.encode_downsample_rate

    def get_decode_upsample_rate(self):
        return self.decode_upsample_rate

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
        """重写 from_pretrained：除了加载主模型权重，
        还要从仓库中额外下载 campplus.onnx 加载到 encoder_xvector_extractor。"""
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
            **kwargs,
        )
        # 从仓库的子目录里查找 campplus.onnx（说话人向量抽取器）
        encoder_xvector_extractor_path = cached_file(
            pretrained_model_name_or_path,
            "campplus.onnx",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if encoder_xvector_extractor_path is None:
            raise ValueError(f"""{pretrained_model_name_or_path}/{encoder_xvector_extractor_path} not exists""")
        model.load_encoder_xvector_extractor(encoder_xvector_extractor_path)

        return model

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, Optional[torch.Tensor]], Qwen3TTSTokenizerV1EncoderOutput]:
        """编码：音频 → (codes, xvectors, ref_mels) 三件套。
        这是 25Hz 版与 12Hz 版的关键区别 —— 同时返回说话人向量与参考 mel，
        供 decode 时作为音色条件。"""
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0
                for *masked*.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 用 padding_mask 把每条音频裁到真实长度（去掉 padding 部分）
        wavs = [value[:mask.sum()] for value, mask in zip(input_values, padding_mask)]

        # 阶段一：音频 → 离散编号（Whisper + GRVQ）
        codes, codes_lens = self.encoder.quantize_speech(wavs)
        # 按真实编号长度裁剪（去掉 batch 内 pad 的部分）
        codes = [c[:l] for c, l in zip(codes, codes_lens)]

        # 阶段二：从同一音频抽取说话人向量 x-vector 和参考 mel
        # 这两样东西在 decode 时作为音色条件输入 DiT
        xvectors = []
        ref_mels = []
        for wav in wavs:
            xvector, ref_mel = self.encoder_xvector_extractor.extract_code(wav.cpu().numpy())
            xvector = torch.tensor(xvector).to(wav.dtype).to(wav.device)
            ref_mel = torch.tensor(ref_mel).to(wav.dtype).to(wav.device)
            xvectors.append(xvector)
            ref_mels.append(ref_mel)

        if not return_dict:
            return (
                codes,
                xvectors,
                ref_mels
            )

        return Qwen3TTSTokenizerV1EncoderOutput(codes, xvectors, ref_mels)

    def decode(
        self,
        audio_codes: torch.Tensor,
        xvectors: torch.Tensor,
        ref_mels: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], Qwen3TTSTokenizerV1DecoderOutput]:
        """解码：(audio_codes + xvectors + ref_mels) 三件套 → 波形。
        与 12Hz 版（只需 codes）不同，25Hz 版必须把说话人向量和参考 mel 一并喂入。"""
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, codes_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            xvectors (`torch.FloatTensor` of shape `(batch_size, xvector_dim)`, *optional*):
                X-vector embeddings computed using `model.encode`.
            ref_mels (`torch.FloatTensor` of shape `(batch_size, mel_length, mel_dim)`, *optional*):
                Reference mel spectrogram computed using `model.encode`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # 用"编号 > -1"的有效位置算出每条样本的实际编号数，再乘上采样倍率得到波形长度
        # 用于最后把生成波形裁到正确长度
        audio_lengths = (audio_codes > -1).sum(1) * self.decode_upsample_rate

        # 把可能存在的 -1（pad 标记）钳到 0，避免 Embedding 越界
        audio_codes = torch.clamp(audio_codes, min=0)
        # 调用 decoder：DiT 采样出 mel + BigVGAN 把 mel 变波形
        audio_values = self.decoder(code=audio_codes,
                                    reference_mel=ref_mels,
                                    conditioning=xvectors)

        # 按真实长度裁剪，去掉多余的尾部样本
        audio_values = [a[:l] for a, l in zip(audio_values, audio_lengths)]

        if not return_dict:
            return (
                audio_values,
            )

        return Qwen3TTSTokenizerV1DecoderOutput(audio_values)


__all__ = ["Qwen3TTSTokenizerV1Model", "Qwen3TTSTokenizerV1PreTrainedModel"]
