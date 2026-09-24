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
"""
Qwen3TTSTokenizerV2 model configuration
========================================

本文件职责（在整体架构中的位置）
--------------------------------
Qwen3-TTS 把语音"分词"成离散编号再用 LLM 生成；语音 tokenizer 负责"音频↔编号"翻译。
本文件定义 **12Hz 版（V2，正式发布）** 语音 tokenizer 的全部配置：
- 每秒约 12.5 步，每步 16 层编号（codebook size = 2048）
- encoder 基于 Mimi（见 `MimiConfig`），decoder 用 Transformer + 卷积 + SnakeBeta
- 顶层配置 `Qwen3TTSTokenizerV2Config` 组合 encoder_config + decoder_config
- 解码器配置 `Qwen3TTSTokenizerV2DecoderConfig` 描述自回归 Transformer + vocoder 结构

术语速查
--------
- codebook 码本：存"标准声音碎片"的字典，共 2048 项
- num_quantizers 量化层数：每步输出 16 个编号（16 层 RVQ）
- RoPE 旋转位置编码：用旋转矩阵给序列编码相对位置
- sliding_window 滑动窗口：局部注意力，限制上下文长度以提高效率
- GQA 分组查询注意力：多个 Q 共享少量 K/V，节省显存
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from transformers import MimiConfig


logger = logging.get_logger(__name__)


class Qwen3TTSTokenizerV2DecoderConfig(PretrainedConfig):
    r"""
    12Hz 版（V2）语音 tokenizer 的 **解码器（code → waveform）** 配置。

    用途
    ----
    定义 V2 解码器结构：自回归 Transformer（处理离散 token）+ vocoder（token → 连续特征 → 波形）。
    与 encoder（基于 Mimi 的 `MimiConfig`）配合，完成"编号→声音"的还原。

    关键参数说明
    -------------
    codebook_size: 每本残差码本的条目数，默认 2048（每步可输出 0~2047 中的一个编号）
    hidden_size: 自回归 Transformer 的隐藏维度（=token 嵌入维度），默认 1024
    latent_dim: 中间 latent 维度，默认 1024
    max_position_embeddings: 解码器能处理的最大序列长度，决定位置嵌入尺寸，默认 8000
    rope_theta: RoPE 旋转位置编码的基频周期，默认 10000（值越大，长距离衰减越慢）
    num_attention_heads: 每层注意力头数，默认 16
    num_key_value_heads: GQA 中 K/V 的头数（≤num_attention_heads），默认 16（此处=Q 头数即退化为 MHA）
    attention_bias: 注意力投影层是否带 bias，默认 False
    sliding_window: 局部注意力的窗口大小，默认 72（只看前后各 72 步，提效且保证因果性）
    intermediate_size: FFN 中间层维度，默认 3072
    hidden_act: FFN 激活函数，默认 "silu"
    layer_scale_initial_scale: LayerScale 初始缩放值，默认 0.01（小初值稳定训练）
    rms_norm_eps: RMSNorm 防除零的 epsilon，默认 1e-5
    num_hidden_layers: 自回归解码器 Transformer 的层数，默认 8
    num_quantizers: vocoder 中 RVQ 的层数（每步输出多少个编号），默认 16
    upsample_rates: 最后波形合成阶段的上采样倍率 (8,5,4,3) → 总倍率 480
    upsampling_ratios: 转置卷积逐步上采样的比率 (2,2)
    decoder_dim: 解码器最后输出层维度，默认 1536
    attention_dropout: 解码器注意力权重 dropout 概率，默认 0.0

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        codebook_size (`int`, *optional*, defaults to 2048):
            Number of entries in each residual codebook used for acoustic token quantization.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the hidden states and embeddings in the autoregressive transformer decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8000):
            Maximum sequence length that the autoregressive decoder can handle. Determines positional embedding size.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period for rotary position embeddings (RoPE) applied to attention layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key and value attention heads used in grouped-query attention (if applicable).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention projection layers.
        sliding_window (`int`, *optional*, defaults to 72):
            Window size for local attention mechanism, limiting attention context to improve efficiency.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the feed-forward (intermediate) layer in each transformer block.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function used in the feed-forward layers. Supports `"silu"`, `"relu"`, `"gelu"`, etc.
        layer_scale_initial_scale (`float`, *optional*, defaults to 0.01):
            Initial value for LayerScale applied in transformer blocks, helping stabilize training.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon value for RMS normalization layers to prevent division by zero.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of transformer blocks in the autoregressive decoder.
        num_quantizers (`int`, *optional*, defaults to 16):
            Number of residual vector quantizers used in the vocoder for fine-grained audio reconstruction.
        upsample_rates (`Tuple[int]`, *optional*, defaults to `(8, 5, 4, 3)`):
            Rate at which features are upsampled in the final waveform synthesis stage.
        upsampling_ratios (`Tuple[int]`, *optional*, defaults to `(2, 2)`):
            Ratios used in transposed convolutional layers to progressively upsample feature maps to waveform.
        decoder_dim (`int`, *optional*, defaults to 1536):
            Final dimensionality of the decoder's output before waveform generation.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability applied to attention weights in the decoder.
    """

    def __init__(
        self,
        codebook_size=2048,
        hidden_size=1024,
        latent_dim=1024,
        max_position_embeddings=8000,
        rope_theta=10000,
        num_attention_heads=16,
        num_key_value_heads=16,
        attention_bias=False,
        sliding_window=72,
        intermediate_size=3072,
        hidden_act="silu",
        layer_scale_initial_scale=0.01,
        rms_norm_eps=1e-5,
        num_hidden_layers=8,
        num_quantizers=16,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        decoder_dim=1536,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 码本大小：每层 RVQ 的字典条目数（V2 正式版为 2048）
        self.codebook_size = codebook_size
        # 自回归 Transformer 的隐藏维度 / token 嵌入维度
        self.hidden_size = hidden_size
        # latent 中间维度
        self.latent_dim = latent_dim
        # 最大位置编号，决定位置嵌入表的大小
        self.max_position_embeddings = max_position_embeddings
        # RoPE 基频（越大长距离衰减越慢）
        self.rope_theta = rope_theta
        # 注意力头数
        self.num_attention_heads = num_attention_heads
        # K/V 头数（GQA；等于 Q 头数时退化为普通多头注意力）
        self.num_key_value_heads = num_key_value_heads
        # 注意力投影是否带 bias
        self.attention_bias = attention_bias
        # 滑动窗口大小（局部注意力，提效 + 保持因果）
        self.sliding_window = sliding_window
        # FFN 中间层维度
        self.intermediate_size = intermediate_size
        # FFN 激活函数
        self.hidden_act = hidden_act
        # LayerScale 初始值（小值稳定深层训练）
        self.layer_scale_initial_scale = layer_scale_initial_scale
        # RMSNorm epsilon
        self.rms_norm_eps = rms_norm_eps
        # Transformer 层数
        self.num_hidden_layers = num_hidden_layers
        # RVQ 量化层数（V2 为 16，即每步输出 16 个编号）
        self.num_quantizers = num_quantizers
        # vocoder 上采样倍率，乘积 = 总上采样倍数
        self.upsample_rates = upsample_rates
        # 转置卷积的上采样比率
        self.upsampling_ratios = upsampling_ratios
        # 解码器输出维度
        self.decoder_dim = decoder_dim
        # 注意力 dropout
        self.attention_dropout = attention_dropout

    @property
    def layer_types(self):
        """
        所有 code→wav 层均采用 sliding_attention（滑动窗口注意力）。

        说明
        ----
        V2 解码器全部使用滑动窗口注意力，统一以局部上下文做自回归生成，
        既保证因果性（只看过去），又把每层注意力复杂度从 O(L^2) 降到 O(L*W)。
        """
        return ["sliding_attention"] * self.num_hidden_layers


class Qwen3TTSTokenizerV2Config(PretrainedConfig):
    """
    12Hz 版（V2）语音 tokenizer **顶层配置**。

    用途
    ----
    组合 encoder_config（基于 Mimi 的 `MimiConfig`，把音频编码为连续特征并 RVQ 量化）
    与 decoder_config（`Qwen3TTSTokenizerV2DecoderConfig`，把 token 解码为波形），
    共同定义 `Qwen3TTSTokenizerV2Model` 的整体结构。

    关键参数
    ---------
    encoder_config: encoder 子模型配置（透传给 `MimiConfig`）
    decoder_config: decoder 子模型配置（透传给 `Qwen3TTSTokenizerV2DecoderConfig`）
    encoder_valid_num_quantizers: encoder 实际使用的有效量化层数（≤ decoder 的 16），默认 16
    input_sample_rate: 输入音频采样率，默认 24000 Hz
    output_sample_rate: 输出音频采样率，默认 24000 Hz
    decode_upsample_rate: 解码端总上采样倍率（编号步 → 波形采样点数），默认 1920
        = 24000 / 12.5 ≈ 1920，对应每秒 12.5 步
    encode_downsample_rate: 编码端总下采样倍率（波形 → 编号步），默认 1920

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_config (`dict`, *optional*): Configuration of the underlying encoder sub-model.
        decoder_config (`dict`, *optional*): Configuration of the underlying decoder sub-model.
    """

    # model_type 用于 transformers 自动注册，对应 12Hz 版（V2）
    model_type = "qwen3_tts_tokenizer_12hz"
    # sub_configs 声明嵌套子配置类，便于 transformers 序列化/反序列化时自动还原
    sub_configs = {
        "encoder_config": MimiConfig,
        "decoder_config": Qwen3TTSTokenizerV2DecoderConfig,
    }

    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        encoder_valid_num_quantizers=16,
        input_sample_rate=24000,
        output_sample_rate=24000,
        decode_upsample_rate=1920,
        encode_downsample_rate=1920,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # encoder_config 缺省时用空 dict，触发 MimiConfig 的默认值
        if encoder_config is None:
            encoder_config = {}
            logger.info("encoder_config is None. Initializing encoder with default values")
        # decoder_config 缺省时用空 dict，触发 DecoderConfig 的默认值
        if decoder_config is None:
            decoder_config = {}
            logger.info("decoder_config is None. Initializing decoder with default values")

        # 用子配置类实例化，将 dict 形态的子配置展开成对象
        self.encoder_config = MimiConfig(**encoder_config)
        self.decoder_config = Qwen3TTSTokenizerV2DecoderConfig(**decoder_config)

        # encoder 实际使用的量化层数（可小于 decoder 的 num_quantizers，用于分层推理）
        self.encoder_valid_num_quantizers = encoder_valid_num_quantizers
        # 输入/输出音频采样率
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        # 解码上采样倍率（一个编号步对应多少采样点），24000/12.5≈1920
        self.decode_upsample_rate = decode_upsample_rate
        # 编码下采样倍率，与上面对称
        self.encode_downsample_rate = encode_downsample_rate


__all__ = ["Qwen3TTSTokenizerV2Config", "Qwen3TTSTokenizerV2DecoderConfig"]
