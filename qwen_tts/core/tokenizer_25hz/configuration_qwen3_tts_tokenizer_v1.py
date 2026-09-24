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
Qwen3TTSTokenizerV1 model configuration
=======================================

本文件职责（在整体架构中的位置）
--------------------------------
定义 **25Hz 版（V1，早期未发布）** 语音 tokenizer 的全部配置。
- 每秒 25 步，每步 1 个编号，码本 32768 项
- encoder 用 Whisper 风格 Transformer + GRVQ 量化（参数见 `Qwen3TTSTokenizerV1EncoderConfig`）
- decoder 用 DiT（扩散 Transformer，生成 mel）+ BigVGAN（声码器，mel → 波形）
  - DiT 参数见 `Qwen3TTSTokenizerV1DecoderDiTConfig`
  - BigVGAN 参数见 `Qwen3TTSTokenizerV1DecoderBigVGANConfig`
- 顶层配置 `Qwen3TTSTokenizerV1Config` 组合 encoder_config + decoder_config
- `Qwen3TTSTokenizerV1DecoderConfig` 进一步组合 dit_config + bigvgan_config

术语速查
--------
- GRVQ 分组残差向量量化：RVQ 的分组变体，把特征先分组再分别做 RVQ
- DiT 扩散 Transformer：用 Transformer 做去噪扩散模型，从噪声生成 mel
- BigVGAN：通用声码器，把 mel 谱图转为波形
- Whisper：OpenAI 语音识别模型，其 encoder 被借用来提取语音特征
- mel 频谱图：声音时间×频率的二维表示
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Qwen3TTSTokenizerV1DecoderDiTConfig(PretrainedConfig):
    r"""
    25Hz 版（V1）解码器的 **DiT（Diffusion Transformer）子模块** 配置。

    用途
    ----
    DiT 是扩散模型 + Transformer 的组合：把离散 token 作为条件，用去噪扩散过程
    逐步把随机噪声变成 mel 频谱图；mel 再交给 BigVGAN 转成波形。
    本配置同时包含 DiT 主体参数和条件 speaker encoder 参数（前缀 enc_*）。

    关键参数（DiT 主体）
    -------------------
    hidden_size: DiT 隐藏维度，默认 1024
    num_hidden_layers: DiT 的 Transformer 层数，默认 22
    num_attention_heads: 每层注意力头数，默认 16
    ff_mult: FFN 隐藏层倍数，默认 2
    emb_dim: 条件/时间步嵌入维度，默认 512
    head_dim: 每个注意力头的维度，默认 64
    repeats: codec 嵌入重复次数（多个量化层映射到同一序列），默认 2
    num_embeds: codec 嵌入表条目数，默认 8193（含特殊 token）
    mel_dim: 输出 mel 谱图的频率维数，默认 80
    dropout: 训练时的 dropout 概率，默认 0.1
    rope_theta / max_position_embeddings: RoPE 基频与最大序列长度
    block_size / look_ahead_layers / look_backward_layers: 滑动窗口分块大小与
        前瞻/后顾层索引（部分层只看未来或过去，混合局部注意力）

    关键参数（speaker encoder，前缀 enc_*）
    --------------------------------------
    enc_emb_dim: 预训练 speaker embedding 输入维度，默认 192
    enc_dim: speaker encoder 输出维度，默认 128
    enc_channels: TDNN/SERes2Net 每层输出通道数，默认 [256,256,256,256,768]
    enc_kernel_sizes: 每层卷积核大小，默认 [5,3,3,3,1]
    enc_dilations: 每层空洞率，默认 [1,2,3,4,1]
    enc_attention_channels: SqueezeExcitation 块的注意力通道数，默认 64
    enc_res2net_scale: Res2Net 块的尺度，默认 2
    enc_se_channels: SE 块 squeeze 后的通道数，默认 64

    This is the configuration class to store the configuration of the Qwen3TTSTokenizerV1DecoderToken2WavDiT.
    It defines the architecture of the DiT model, which is used for generating mel-spectrograms from tokens.

    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            The dimension of the model.
        num_hidden_layers (`int`, *optional*, defaults to 22):
            The number of transformer blocks in the DiT model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of attention heads in each transformer block.
        ff_mult (`int`, *optional*, defaults to 2):
            The multiplier for the feedforward layer in each transformer block.
        emb_dim (`int`, *optional*, defaults to 512):
            The dimension of the embedding layer.
        head_dim (`int`, *optional*, defaults to 64):
            The dimension of each attention head.
        repeats (`int`, *optional*, defaults to 2):
            The number of times the codec embeddings are repeated.
        num_embeds (`int`, *optional*, defaults to 8193):
            The number of unique embeddings in the codec.
        mel_dim (`int`, *optional*, defaults to 80):
            The dimension of the mel-spectrogram.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout rate for the transformer blocks.

        enc_emb_dim (`int`, *optional*, defaults to 192):
            The dimension of the pre-trained speaker embedding.
        enc_dim (`int`, *optional*, defaults to 128):
            The dimension of the encoder output.
        enc_channels (`list[int]`, *optional*, defaults to `[256, 256, 256, 256, 768]`):
            A list of output channels for each TDNN/SERes2Net layer in the encoder.
        enc_kernel_sizes (`list[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
            A list of kernel sizes for each layer in the encoder.
        enc_dilations (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
            A list of dilations for each layer in the encoder.
        enc_attention_channels (`int`, *optional*, defaults to 64):
            The number of attention channels in the SqueezeExcitationBlock.
        enc_res2net_scale (`int`, *optional*, defaults to 2):
            The scale of the Res2Net block in the encoder.
        enc_se_channels (`int`, *optional*, defaults to 64):
            The number of output channels after squeeze in the SqueezeExcitationBlock.
    """

    model_type = "qwen3_tts_tokenizer_v1_decoder_dit"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=22,
        num_attention_heads=16,
        ff_mult=2,
        emb_dim=512,
        head_dim=64,
        rope_theta=10000.0,
        max_position_embeddings=32768,
        block_size=24,
        look_ahead_layers=[10],
        look_backward_layers=[0, 20],
        repeats=2,
        num_embeds=8193,
        mel_dim=80,
        dropout=0.1,
        enc_emb_dim=192,
        enc_dim=128,
        enc_channels=[256, 256, 256, 256, 768],
        enc_kernel_sizes=[5, 3, 3, 3, 1],
        enc_dilations=[1, 2, 3, 4, 1],
        enc_attention_channels=64,
        enc_res2net_scale=2,
        enc_se_channels=64,
        **kwargs,
    ):
        # === DiT 主体参数 ===
        self.hidden_size = hidden_size  # 隐藏维度
        self.num_hidden_layers = num_hidden_layers  # Transformer 层数
        self.num_attention_heads = num_attention_heads  # 注意力头数
        self.ff_mult = ff_mult  # FFN 倍数
        self.emb_dim = emb_dim  # 嵌入维度
        self.head_dim = head_dim  # 每个头维度
        self.rope_theta = rope_theta  # RoPE 基频
        self.max_position_embeddings = max_position_embeddings  # 最大序列长度
        self.block_size = block_size  # 滑动窗口分块大小
        self.look_ahead_layers = look_ahead_layers  # 前瞻层索引（看未来帧）
        self.look_backward_layers = look_backward_layers  # 后顾层索引（看过往帧）
        self.repeats = repeats  # codec 嵌入重复次数
        self.num_embeds = num_embeds  # codec 嵌入表大小（含特殊 token）
        self.mel_dim = mel_dim  # mel 谱图频率维
        self.dropout = dropout  # dropout 概率
        # === speaker encoder 参数（条件输入）===
        self.enc_emb_dim = enc_emb_dim  # 预训练 speaker embedding 输入维
        self.enc_dim = enc_dim  # speaker encoder 输出维
        self.enc_channels = enc_channels  # 每层通道数
        self.enc_kernel_sizes = enc_kernel_sizes  # 每层卷积核
        self.enc_dilations = enc_dilations  # 每层空洞率
        self.enc_attention_channels = enc_attention_channels  # SE 注意力通道数
        self.enc_res2net_scale = enc_res2net_scale  # Res2Net 尺度
        self.enc_se_channels = enc_se_channels  # SE squeeze 通道数
        super().__init__(**kwargs)


class Qwen3TTSTokenizerV1DecoderBigVGANConfig(PretrainedConfig):
    r"""
    25Hz 版（V1）解码器的 **BigVGAN 声码器子模块** 配置。

    用途
    ----
    BigVGAN 是通用神经声码器，把 DiT 生成的 mel 频谱图转换为最终音频波形。
    主要由"上采样卷积栈 + 多残差块（resblock）"组成，逐级上采样到目标采样率。

    关键参数
    ---------
    mel_dim: 输入 mel 的频率维数，默认 80（与 DiT 输出对齐）
    upsample_initial_channel: 第一层上采样卷积的通道数，默认 1536（越宽模型越强）
    resblock_kernel_sizes: 每个残差块的卷积核大小，默认 [3,7,11]（多尺度感受野）
    resblock_dilation_sizes: 每个残差块的空洞率列表，默认 [[1,3,5],[1,3,5],[1,3,5]]
    upsample_rates: 每级上采样倍率，默认 [5,3,2,2,2,2]，乘积=240（mel 帧率→采样率）
    upsample_kernel_sizes: 每级上采样卷积核大小，默认 [11,7,4,4,4,4]

    This is the configuration class to store the configuration of the Qwen3TTSTokenizerV1DecoderToken2WavBigVGAN module.
    It defines the architecture of the BigVGAN model, which is used for converting mel-spectrograms to waveforms.

    Args:
        mel_dim (`int`, *optional*, defaults to 80):
            The dimension of the mel-spectrogram.
        upsample_initial_channel (`int`, *optional*, defaults to 1536):
            The number of channels in the initial upsampling layer.
        resblock_kernel_sizes (`list[int]`, *optional*, defaults to `[3, 7, 11]`):
            A list of kernel sizes for each residual block.
        resblock_dilation_sizes (`list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A list of dilation sizes for each residual block.
        upsample_rates (`list[int]`, *optional*, defaults to `[5, 3, 2, 2, 2, 2]`):
            A list of upsampling rates for each upsampling layer.
        upsample_kernel_sizes (`list[int]`, *optional*, defaults to `[11, 7, 4, 4, 4, 4]`):
            A list of kernel sizes for each upsampling layer.
    """

    model_type = "qwen3_tts_tokenizer_v1_decoder_bigvgan"

    def __init__(
        self,
        mel_dim=80,
        upsample_initial_channel=1536,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[5, 3, 2, 2, 2, 2],
        upsample_kernel_sizes=[11, 7, 4, 4, 4, 4],
        **kwargs,
    ):
        self.mel_dim = mel_dim  # 输入 mel 频率维
        self.upsample_initial_channel = upsample_initial_channel  # 起始上采样通道数
        self.resblock_kernel_sizes = resblock_kernel_sizes  # 残差块卷积核
        self.resblock_dilation_sizes = resblock_dilation_sizes  # 残差块空洞率
        self.upsample_rates = upsample_rates  # 各级上采样倍率（乘积=总倍率）
        self.upsample_kernel_sizes = upsample_kernel_sizes  # 各级上采样卷积核
        super().__init__(**kwargs)


class Qwen3TTSTokenizerV1DecoderConfig(PretrainedConfig):
    r"""
    25Hz 版（V1）解码器 **顶层配置**：组合 DiT + BigVGAN。

    用途
    ----
    把 `dit_config`（生成 mel）与 `bigvgan_config`（mel → 波形）拼成完整 token2wav 解码器。
    是 `Qwen3TTSTokenizerV1Config` 的子配置。

    关键参数
    ---------
    dit_config: DiT 子模块配置（透传给 `Qwen3TTSTokenizerV1DecoderDiTConfig`）
    bigvgan_config: BigVGAN 子模块配置（透传给 `Qwen3TTSTokenizerV1DecoderBigVGANConfig`）

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        dit_config ([`DiT_Args`], *optional*):
            Configuration class for the Diffusion Transformer (DiT) module responsible for generating mel-spectrograms.
        bigvgan_config ([`BigVGAN_Args`], *optional*):
            Configuration class for the BigVGAN module responsible for converting mel-spectrograms to waveforms.
    """

    model_type = "qwen3_tts_tokenizer_v1_decoder"
    sub_configs = {
        "dit_config": Qwen3TTSTokenizerV1DecoderDiTConfig,
        "bigvgan_config": Qwen3TTSTokenizerV1DecoderBigVGANConfig,
    }

    def __init__(self, dit_config=None, bigvgan_config=None, **kwargs):
        # dit_config 缺省用空 dict，触发 DiTConfig 默认值
        if dit_config is None:
            dit_config = {}
        # bigvgan_config 缺省用空 dict，触发 BigVGANConfig 默认值
        if bigvgan_config is None:
            bigvgan_config = {}
        # 用子配置类展开 dict 形态子配置
        self.dit_config = Qwen3TTSTokenizerV1DecoderDiTConfig(**dit_config)
        self.bigvgan_config = Qwen3TTSTokenizerV1DecoderBigVGANConfig(**bigvgan_config)
        super().__init__(**kwargs)


class Qwen3TTSTokenizerV1EncoderConfig(PretrainedConfig):
    r"""
    25Hz 版（V1）**编码器** 配置：Whisper 风格 Transformer + GRVQ 量化。

    用途
    ----
    encoder 取 mel 频谱图 → 经 Whisper 风格 Transformer 提取高层音频表示 →
    由 GRVQ（分组残差向量量化）把连续特征离散成编号。每步输出 1 个编号，码本 32768 项。

    关键参数（Whisper 主体）
    ------------------------
    n_mels: 输入 mel 频率维，默认 128
    n_ctx: 最大序列长度（帧/token），默认 1500
    n_state: encoder Transformer 隐藏维度，默认 1280
    n_head: 注意力头数，默认 20
    n_layer: Transformer 层数，默认 32
    n_window: 局部注意力/分块窗口大小，默认 100
    output_dim: encoder 输出维度（投影后），默认 3584（与 LLM 对齐）

    关键参数（训练/并行）
    ---------------------
    grad_checkpointing: 是否启用梯度检查点（省显存换计算），默认 False
    enable_mp: 是否启用模型并行，默认 False
    audio_sequence_parallel: 是否启用音频分支序列并行，默认 False

    关键参数（VQ 模块）
    -------------------
    audio_vq_type: VQ 类型，默认 "GRVQ"（分组残差 VQ）
    audio_vq_layers: VQ 层数（残差量化器个数），默认 6
    audio_vq_codebook_size: 每本码本条目数，默认 32768
    audio_vq_codebook_dim: 码本向量维度，默认 1280（常等于 encoder 隐藏维）
    audio_vq_pe: VQ 模块内部是否使用位置编码，默认 True
    audio_vq_ds_rate: VQ 前的时间下采样倍率，默认 2

    The encoder typically takes mel-spectrogram features and produces high-level audio representations, then (optionally)
    applies an Audio-VQ module (e.g., GRVQ) to discretize continuous representations into codes.

    Args:
        n_mels (`int`, *optional*, defaults to 128):
            Number of mel bins in the input mel-spectrogram.
        n_ctx (`int`, *optional*, defaults to 1500):
            Maximum input sequence length (in frames/tokens) for the encoder.
        n_state (`int`, *optional*, defaults to 1280):
            Hidden size (model dimension) of the encoder transformer.
        n_head (`int`, *optional*, defaults to 20):
            Number of attention heads in each transformer layer.
        n_layer (`int`, *optional*, defaults to 32):
            Number of transformer layers.
        n_window (`int`, *optional*, defaults to 100):
            Window size used by the model for local attention / chunking (implementation-dependent).
        output_dim (`int`, *optional*, defaults to 3584):
            Output feature dimension produced by the encoder head (before/after projection, implementation-dependent).

        grad_checkpointing (`bool`, *optional*, defaults to `False`):
            Whether to enable gradient checkpointing to reduce memory usage during training.
        enable_mp (`bool`, *optional*, defaults to `False`):
            Whether to enable model parallel features (implementation-dependent).
        audio_sequence_parallel (`bool`, *optional*, defaults to `False`):
            Whether to enable sequence parallelism for audio branch (implementation-dependent).

        audio_vq_type (`str`, *optional*, defaults to `"GRVQ"`):
            Type of audio vector-quantization module. Common choices: `"GRVQ"`, `"RVQ"`, etc.
        audio_vq_layers (`int`, *optional*, defaults to 6):
            Number of VQ layers / quantizers (e.g., number of residual quantizers for RVQ/GRVQ-like designs).
        audio_vq_codebook_size (`int`, *optional*, defaults to 32768):
            Size of each codebook (number of entries).
        audio_vq_codebook_dim (`int`, *optional*, defaults to 1280):
            Dimension of codebook vectors (often equals encoder hidden size).
        audio_vq_pe (`bool`, *optional*, defaults to `True`):
            Whether to use positional encoding (or position embeddings) inside the VQ module.
        audio_vq_ds_rate (`int`, *optional*, defaults to 2):
            Downsampling rate applied before VQ (e.g., temporal downsample factor).
    """

    model_type = "qwen3_tts_tokenizer_v1_encoder"

    def __init__(
        self,
        n_mels=128,
        n_ctx=1500,
        n_state=1280,
        n_head=20,
        n_layer=32,
        n_window=100,
        output_dim=3584,
        grad_checkpointing=False,
        enable_mp=False,
        audio_sequence_parallel=False,
        audio_vq_type="GRVQ",
        audio_vq_layers=6,
        audio_vq_codebook_size=32768,
        audio_vq_codebook_dim=1280,
        audio_vq_pe=True,
        audio_vq_ds_rate=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # === Whisper 主体参数 ===
        self.n_mels = n_mels  # 输入 mel 频率维
        self.n_ctx = n_ctx  # 最大序列长度
        self.n_state = n_state  # 隐藏维度
        self.n_head = n_head  # 注意力头数
        self.n_layer = n_layer  # Transformer 层数
        self.n_window = n_window  # 局部注意力窗口
        self.output_dim = output_dim  # 输出维度（投影后）
        # === 训练 / 并行参数 ===
        self.grad_checkpointing = grad_checkpointing  # 梯度检查点
        self.enable_mp = enable_mp  # 模型并行
        self.audio_sequence_parallel = audio_sequence_parallel  # 音频序列并行
        # === VQ 模块参数 ===
        self.audio_vq_type = audio_vq_type  # VQ 类型（GRVQ）
        self.audio_vq_layers = audio_vq_layers  # VQ 层数
        self.audio_vq_codebook_size = audio_vq_codebook_size  # 码本条目数
        self.audio_vq_codebook_dim = audio_vq_codebook_dim  # 码本向量维度
        self.audio_vq_pe = audio_vq_pe  # VQ 内是否加位置编码
        self.audio_vq_ds_rate = audio_vq_ds_rate  # VQ 前下采样倍率


class Qwen3TTSTokenizerV1Config(PretrainedConfig):
    """
    25Hz 版（V1）语音 tokenizer **顶层配置**。

    用途
    ----
    组合 encoder_config（`Qwen3TTSTokenizerV1EncoderConfig`：Whisper + GRVQ，把音频量化成编号）
    与 decoder_config（`Qwen3TTSTokenizerV1DecoderConfig`：DiT + BigVGAN，把编号还原成波形），
    共同定义 `Qwen3TTSTokenizerV1Model` 的整体结构。

    关键参数
    ---------
    encoder_config: encoder 子模型配置
    decoder_config: decoder 子模型配置
    input_sample_rate: 输入音频采样率，默认 24000 Hz
    output_sample_rate: 输出音频采样率，默认 24000 Hz
    decode_upsample_rate: 解码上采样倍率（编号步→采样点数），默认 1920
    encode_downsample_rate: 编码下采样倍率，默认 1920

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        encoder_config (`dict`, *optional*): Configuration of the underlying encoder sub-model.
        decoder_config (`dict`, *optional*): Configuration of the underlying decoder sub-model.
    """

    # model_type 对应 25Hz 版（V1）
    model_type = "qwen3_tts_tokenizer_25hz"
    # sub_configs 声明子配置类，便于自动序列化/反序列化
    sub_configs = {
        "encoder_config": Qwen3TTSTokenizerV1EncoderConfig,
        "decoder_config": Qwen3TTSTokenizerV1DecoderConfig,
    }

    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        input_sample_rate=24000,
        output_sample_rate=24000,
        decode_upsample_rate=1920,
        encode_downsample_rate=1920,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # encoder_config 缺省用空 dict 触发默认值
        if encoder_config is None:
            encoder_config = {}
            logger.info("encoder_config is None. Initializing encoder with default values")
        # decoder_config 缺省用空 dict 触发默认值
        if decoder_config is None:
            decoder_config = {}
            logger.info("decoder_config is None. Initializing decoder with default values")

        # 展开子配置为对象
        self.encoder_config = Qwen3TTSTokenizerV1EncoderConfig(**encoder_config)
        self.decoder_config = Qwen3TTSTokenizerV1DecoderConfig(**decoder_config)

        # 输入/输出音频采样率
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        # 解码上采样倍率（每编号步对应多少采样点）
        self.decode_upsample_rate = decode_upsample_rate
        # 编码下采样倍率（与上面对称）
        self.encode_downsample_rate = encode_downsample_rate


__all__ = [
    "Qwen3TTSTokenizerV1Config", 
    "Qwen3TTSTokenizerV1EncoderConfig",
    "Qwen3TTSTokenizerV1DecoderConfig", 
    "Qwen3TTSTokenizerV1DecoderBigVGANConfig",
    "Qwen3TTSTokenizerV1DecoderDiTConfig"
]
