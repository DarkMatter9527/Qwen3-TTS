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
from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)

# =============================================================================
# 文件总览：Qwen3-TTS 配置类定义
# -----------------------------------------------------------------------------
# Qwen3-TTS 是端到端文本转语音(TTS)大模型。本文件定义模型的所有"旋钮"（超参数），
# 决定模型大小、层数、维度、词表大小等关键结构信息。
#
# 配置类层级关系（自上而下组合）：
#   Qwen3TTSConfig（顶层）
#     ├── talker_config: Qwen3TTSTalkerConfig（主 talker，Qwen3 风格 Transformer）
#     │     └── code_predictor_config: Qwen3TTSTalkerCodePredictorConfig（sub-talker 小模型）
#     └── speaker_encoder_config: Qwen3TTSSpeakerEncoderConfig（说话人编码器 ECAPA-TDNN）
#
# 顶层还包含：tokenizer_type(25hz/12hz)、tts_model_size(0.6B/1.7B)、
# tts_model_type(base/custom_voice/voice_design) 以及各类特殊 token id。
#
# 关键术语通俗解释：
#   - 配置/超参数：模型的"旋钮"，决定模型大小、层数、维度等
#   - vocab_size 词表大小：模型能输出的不同 token 数
#   - hidden_size 隐藏维度：模型内部向量的长度
#   - num_hidden_layers 层数：Transformer 堆了多少层
#   - num_attention_heads 注意力头数：注意力并行分多少个头
#   - num_key_value_heads (GQA)：多个查询头共享一组 key/value 头，省显存
#   - head_dim：每个注意力头的维度
#   - RoPE 旋转位置编码：用旋转把位置编进向量
#   - sliding_window 滑动窗口：只看局部窗口的注意力，省显存
#   - intermediate_size：MLP 中间层维度
#   - num_code_groups=16：多码本，每个时间步用 16 个编号表示音频
#   - codec_eos/bos/pad/think 等 token id：控制标记的编号
#   - spk_id：每个预置音色对应的 id 字典
#   - ECAPA-TDNN：提取说话人特征的网络
#   - TDNN 时延神经网络 / Res2Net 多尺度残差 / SE 通道注意力 / AttentiveStatsPooling 注意力统计池化
#   - PretrainedConfig：HuggingFace 配置基类，提供序列化、加载等通用能力
# =============================================================================


class Qwen3TTSSpeakerEncoderConfig(PretrainedConfig):
    # -----------------------------------------------------------------------------
    # 说话人编码器配置（ECAPA-TDNN 架构）
    # -----------------------------------------------------------------------------
    # 用途：从一段参考音频中提取"说话人特征向量"（d-vector），用于音色克隆。
    #       在 custom_voice / voice_design 模式下，把参考音频喂给此编码器，
    #       得到的向量作为 talker 的条件输入，使合成语音模仿目标音色。
    # 架构：ECAPA-TDNN = TDNN 时延神经网络 + Res2Net 多尺度残差 + SE 通道注意力
    #       + AttentiveStatisticsPooling 注意力统计池化，最后线性投影到 enc_dim。
    # 关键参数：
    #   mel_dim         输入 mel 频谱图的维度（一帧音频的能量分布）
    #   enc_dim         最终说话人嵌入向量的维度
    #   enc_channels    各 TDNN/SE-Res2Net 层的输出通道数列表
    #   enc_dilations   各层膨胀系数，扩大感受野（看更远的历史帧）
    #   sample_rate     音频采样率（默认 24000Hz）
    # -----------------------------------------------------------------------------
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSSpeakerEncoder`].
    It is used to instantiate a Qwen3TTS speaker encoder model according to the specified arguments, defining the model
    architecture. The architecture is based on the ECAPA-TDNN model.

    Args:
        mel_dim (`int`, *optional*, defaults to 128):
            The dimension of the input mel-spectrogram.
            （中文：输入 mel 频谱图的维度，即一帧音频经傅里叶变换后保留多少个频率桶；
             常见值 80/128，越大频率分辨率越高）
        enc_dim (`int`, *optional*, defaults to 192):
            The dimension of the final speaker embedding.
            （中文：最终输出说话人嵌入向量的维度，决定音色特征的表达能力；
             通常 192~512，越大表达能力越强但更耗资源）
        enc_channels (`list[int]`, *optional*, defaults to `[512, 512, 512, 512, 1536]`):
            A list of output channels for each TDNN/SERes2Net layer in the encoder. The first channel size is for the initial TDNN layer,
            the intermediate ones for the `SqueezeExcitationRes2NetBlock` layers, and the last one for the multi-layer feature aggregation.
            （中文：编码器每一层的输出通道数列表。第 1 个是初始 TDNN 层，
             中间几个是 SE-Res2Net 残差块，最后一个是多层特征聚合层；
             通道数越大表示该层学到的特征越丰富）
        enc_kernel_sizes (`list[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
            A list of kernel sizes for each layer in the encoder, corresponding to `enc_channels`.
            （中文：每一层卷积核的尺寸列表，与 enc_channels 一一对应；
             核越大感受野越广，但计算量也越大）
        enc_dilations (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
            A list of dilations for each layer in the encoder, corresponding to `enc_channels`.
            （中文：每一层的膨胀系数。膨胀卷积在卷积核元素间插入空洞，
             1/2/3/4 逐层扩大感受野，让网络看到更长时程的语音上下文）
        enc_attention_channels (`int`, *optional*, defaults to 128):
            The number of attention channels in the `AttentiveStatisticsPooling` layer.
            （中文：注意力统计池化层的注意力通道数。该层用注意力权重对时序特征
             做加权求和与方差，把变长时序压缩成定长说话人向量）
        enc_res2net_scale (`int`, *optional*,defaults to 8):
            The scale of the `Res2NetBlock` in the encoder.
            （中文：Res2Net 多尺度残差块的分组尺度。把一个卷积拆成多组串联小卷积，
             增加感受野的多尺度多样性，scale 越大分组越多）
        enc_se_channels (`int`, *optional*, defaults to 128):
            The number of channels in the squeeze part of the `SqueezeExcitationBlock`.
            （中文：SE 通道注意力块的压缩通道数。SE 块先压缩再扩展，
             给每个通道学习一个 0~1 的权重，相当于让网络自动选择重要特征通道）
        sample_rate (`int`, *optional*, defaults to 24000):
            （中文：音频采样率，单位 Hz。24000 表示每秒 24000 个采样点；
             提取 mel 频谱时按此率分帧，必须与训练/推理音频的采样率一致）
    """
    def __init__(
        self,
        mel_dim=128,
        enc_dim=1024,
        enc_channels=[512, 512, 512, 512, 1536],
        enc_kernel_sizes=[5, 3, 3, 3, 1],
        enc_dilations=[1, 2, 3, 4, 1],
        enc_attention_channels=128,
        enc_res2net_scale=8,
        enc_se_channels=128,
        sample_rate=24000,
    ):
        self.mel_dim = mel_dim
        self.enc_dim = enc_dim
        self.enc_channels = enc_channels
        self.enc_kernel_sizes = enc_kernel_sizes
        self.enc_dilations = enc_dilations
        self.enc_attention_channels = enc_attention_channels
        self.enc_res2net_scale = enc_res2net_scale
        self.enc_se_channels = enc_se_channels
        self.sample_rate = sample_rate


class Qwen3TTSTalkerCodePredictorConfig(PretrainedConfig):
    # -----------------------------------------------------------------------------
    # sub-talker 代码预测器配置（小模型）
    # -----------------------------------------------------------------------------
    # 用途：作为 talker 的子模块（code_predictor），负责在多码本(num_code_groups)场景下
    #       预测每个时间步的多个 codec 编号。它是一个轻量级 Qwen3 风格 Transformer
    #       （默认仅 5 层、hidden 1024），挂在主 talker 之上做并行/迭代解码。
    # 与主 talker 关系：主 talker 给出粗粒度文本条件，code_predictor 在此基础上
    #       细化出每一步的多码本编号，从而合成高保真音频。
    # 关键参数：
    #   vocab_size           小模型词表大小（编号空间）
    #   hidden_size          小模型隐藏维度
    #   num_hidden_layers    小模型层数（默认 5，远小于主 talker）
    #   num_code_groups      多码本组数（默认 32），每步预测这么多编号
    #   rope/sliding_window  位置编码与滑动窗口，与主 talker 类似
    # -----------------------------------------------------------------------------
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSTalkerCodePredictorModel`]. It is used to instantiate a
    Qwen3TTSTalkerCodePredictor model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3TTSTalkerCodePredictor model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3TTSTalkerCodePredictorModel`]
            （中文：词表大小，即模型能输出的不同 token 数。对子模型而言这是
             codec 编号的取值范围；值越大可表达越多种编号）
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
            （中文：隐藏维度，模型内部向量的长度。越大表达能力越强，但更耗显存）
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
            （中文：MLP 前馈网络的中间层维度，通常是 hidden_size 的若干倍，
             决定每次变换能混合多少信息）
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
            （中文：Transformer 堆叠的层数。层数越多模型越深、容量越大）
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
            （中文：每层注意力的并行头数。多头注意力让模型在不同子空间同时关注不同位置）
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
            （中文：GQA 分组查询注意力中的 key/value 头数。多个查询头共享一组
             key/value 头，能在几乎不掉点的情况下显著节省 KV 缓存显存。
             等于 num_attention_heads 即普通多头注意力，等于 1 即多查询注意力）
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
            （中文：每个注意力头的维度。head_dim * num_attention_heads 通常≈hidden_size）
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
            （中文：非线性激活函数。silu = x*sigmoid(x)，比 relu 更平滑）
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
            （中文：模型能处理的最大序列长度。超过需要外推或 rope 缩放）
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            （中文：权重初始化用截断正态分布的标准差，影响训练初期稳定性）
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
            （中文：RMSNorm 归一化层的防除零小量。RMSNorm 用均方根归一化，比 LayerNorm 更省算力）
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
            （中文：是否缓存上一轮的 key/value，加速自回归生成。推理时通常开）
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
            （中文：是否让输入嵌入矩阵与输出投影矩阵共享权重。共享可省参数，
             小模型常用，大模型一般不共享）
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
            （中文：RoPE 旋转位置编码的基准周期。越大对长距离外推越友好）
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            （中文：RoPE 缩放配置字典。用于把训练时长外推到更长序列。
             常见子字段：rope_type 缩放类型(default/linear/dynamic/yarn/longrope/llama3)、
             factor 缩放倍数、original_max_position_embeddings 训练时最大长度、
             attention_factor 注意力缩放、beta_fast/beta_slow yarn 的外推/插值边界、
             short_factor/long_factor longrope 的短/长上下文缩放因子、
             low_freq_factor/high_freq_factor llama3 的低/高频分量缩放）
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
            （中文：注意力 Q/K/V/输出投影是否加偏置。Qwen 系列默认不加，省参数）
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
            （中文：是否启用滑动窗口注意力。开启后部分层只看局部窗口，省显存、加速长序列）
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
            （中文：滑动窗口大小（token 数）。窗口越大看的上下文越多但越耗显存）
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers using full attention. The first `max_window_layers` layers will use full attention, while any
            additional layer afterwards will use SWA (Sliding Window Attention).
            （中文：使用全注意力（非滑动窗口）的层数。前 N 层用全注意力，
             之后的层切换到滑动窗口注意力，兼顾全局与局部）
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
            （中文：每层注意力类型列表，逐层指定 full_attention 或 sliding_attention）
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
            （中文：注意力权重的 dropout 比例，防过拟合。推理时一般设为 0）
        num_code_groups (`int`, *optional*, defaults to 32):
            （中文：多码本组数。每个时间步预测 num_code_groups 个编号，
             共同表示该帧音频；组数越多音频细节越丰富，但计算量也越大）

    """

    model_type = "qwen3_tts_talker_code_predictor"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3TTSTalkerCodePredictor`
    # 张量并行(TP)切分方案：把每个线性层按列或行切到多张卡上并行计算。
    #   colwise 列切：权重按输出维度切，各卡算部分输出再拼接（q/k/v/gate/up）
    #   rowwise 行切：权重按输入维度切，各卡算部分结果再相加（o/down）
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    # 流水线并行(PP)切分方案：把模型按阶段分到多卡，各阶段定义输入/输出张量名。
    #   embed_tokens  嵌入层：input_ids → inputs_embeds
    #   layers        堆叠层：hidden_states + attention_mask → hidden_states
    #   norm          最终归一化：hidden_states → hidden_states
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=2048,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=5,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0,
        num_code_groups=32,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        # 仅在启用滑动窗口时才记录窗口大小，否则置 None（表示该层用全注意力）
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        # 向后兼容：旧配置可能不传 num_key_value_heads，此时退化为普通多头注意力
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # 校验 RoPE 旋转位置编码参数合法性
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # 向后兼容：旧字段名 'type' 迁移为 'rope_type'
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        # 若未显式指定每层注意力类型，则按规则自动生成：
        # 前 max_window_layers 层用全注意力，其后的层用滑动窗口注意力
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)
        # 多码本组数：每个时间步预测这么多编号，共同表示一帧音频
        self.num_code_groups = num_code_groups


class Qwen3TTSTalkerConfig(PretrainedConfig):
    # -----------------------------------------------------------------------------
    # 主 Talker 配置（Qwen3 风格 Transformer）
    # -----------------------------------------------------------------------------
    # 用途：Qwen3-TTS 的核心生成模块。接收文本条件、说话人特征与语言/方言信息，
    #       自回归地预测每一步的 codec 编号，再交给声码器还原为音频波形。
    #       本质是一个 Qwen3 风格的 LLM，但输出的是音频 codec token 而非文字。
    # 架构：标准的因果 Transformer（嵌入→堆叠层→归一化→输出），
    #       每层含 GQA 自注意力 + SwiGLU MLP + RMSNorm + RoPE。
    # 关键参数：
    #   vocab_size           主模型词表大小（含 codec token 与文本 token）
    #   hidden_size/intermediate_size  主模型维度
    #   num_hidden_layers    主模型层数（默认 20，远多于子模型）
    #   num_attention_heads/num_key_value_heads  GQA 头配置
    #   text_hidden_size      文本侧编码维度，用于对齐文本与音频隐空间
    #   num_code_groups       多码本组数（默认 32）
    #   codec_eos/bos/pad/think_id 等  控制标记的编号，决定生成流程
    #   spk_id                说话人字典，预置音色 id→说话人信息
    #   codec_language_id     语言字典，预置语言 id→语言信息
    # -----------------------------------------------------------------------------
    r"""
    This is the configuration class to store the configuration of a [`Qwen3TTSTalkerModel`]. It is used to instantiate a
    Qwen3TTSTalker model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3TTSTalker model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3TTSTalkerModel`]
            （中文：词表大小，即模型能输出的不同 token 数。talker 的词表同时包含
             文本 token 和 codec 音频 token）
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
            （中文：隐藏维度，模型内部向量的长度。越大表达能力越强，但更耗显存）
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimension of the MLP representations.
            （中文：MLP 前馈网络的中间层维度，通常是 hidden_size 的若干倍）
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
            （中文：Transformer 堆叠的层数。层数越多模型越深、容量越大）
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
            （中文：每层注意力的并行头数。多头注意力让模型在不同子空间同时关注不同位置）
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
            （中文：GQA 分组查询注意力中的 key/value 头数。多个查询头共享一组
             key/value 头，能在几乎不掉点的情况下显著节省 KV 缓存显存。
             等于 num_attention_heads 即普通多头注意力，等于 1 即多查询注意力）

        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
            （中文：非线性激活函数。silu = x*sigmoid(x)，比 relu 更平滑）
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
            （中文：模型能处理的最大序列长度。超过需要外推或 rope 缩放）
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            （中文：权重初始化用截断正态分布的标准差，影响训练初期稳定性）
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
            （中文：RMSNorm 归一化层的防除零小量。RMSNorm 用均方根归一化，比 LayerNorm 更省算力）
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
            （中文：是否缓存上一轮的 key/value，加速自回归生成。推理时通常开）
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
            （中文：是否让输入嵌入矩阵与输出投影矩阵共享权重。共享可省参数）
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
            （中文：RoPE 旋转位置编码的基准周期。越大对长距离外推越友好）
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            （中文：RoPE 缩放配置字典。用于把训练时长外推到更长序列。
             常见子字段：rope_type 缩放类型、factor 缩放倍数、original_max_position_embeddings
             训练时最大长度、attention_factor 注意力缩放、beta_fast/beta_slow yarn 的外推/插值边界、
             short_factor/long_factor longrope 的短/长上下文缩放因子、
             low_freq_factor/high_freq_factor llama3 的低/高频分量缩放）
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
            （中文：注意力 Q/K/V/输出投影是否加偏置。Qwen 系列默认不加，省参数）
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
            （中文：是否启用滑动窗口注意力。开启后部分层只看局部窗口，省显存、加速长序列）
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
            （中文：滑动窗口大小（token 数）。窗口越大看的上下文越多但越耗显存）
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
            （中文：注意力权重的 dropout 比例，防过拟合。推理时一般设为 0）
    """

    model_type = "qwen3_tts_talker"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3TTSTalker`
    # 张量并行(TP)切分方案：colwise 列切/rowwise 行切，与子模型同理
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    # 流水线并行(PP)切分方案：按 embed_tokens / layers / norm 三个阶段划分
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    # 子配置声明：code_predictor_config 字段会被解析为 Qwen3TTSTalkerCodePredictorConfig
    sub_configs = {"code_predictor_config": Qwen3TTSTalkerCodePredictorConfig}

    def __init__(
        self,
        code_predictor_config=None,
        vocab_size=3072,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        attention_dropout=0,
        num_code_groups=32,
        text_hidden_size=2048,
        codec_eos_token_id=4198,
        codec_think_id=4202,
        codec_nothink_id=4203,
        codec_think_bos_id=4204,
        codec_think_eos_id=4205,
        codec_pad_id=4196,
        codec_bos_id=4197,
        spk_id=None,
        spk_is_dialect=None,
        codec_language_id=None,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        # 仅在启用滑动窗口时才记录窗口大小，否则置 None（表示用全注意力）
        self.sliding_window = sliding_window if use_sliding_window else None

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # 校验 RoPE 旋转位置编码参数合法性
        # BC: if there is a 'type' field, move it to 'rope_type'.
        # 向后兼容：旧字段名 'type' 迁移为 'rope_type'
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        # 子模型 code_predictor 的配置：支持三种传入方式——
        # 1) None：用默认配置实例化一个子模型
        # 2) 已是 Qwen3TTSTalkerCodePredictorConfig 实例：直接使用
        # 3) 字典：用字典参数实例化子模型配置
        if code_predictor_config is None:
            code_predictor_config = {}
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig()
            logger.info("code_predictor_config is None. Initializing code_predictor model with default values")
        elif isinstance(code_predictor_config, Qwen3TTSTalkerCodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig(**code_predictor_config)
        # 多码本组数：每个时间步预测这么多编号，共同表示一帧音频
        self.num_code_groups = num_code_groups
        # 文本侧编码维度，用于把文本特征投影到与音频隐空间对齐的维度
        self.text_hidden_size = text_hidden_size
        # 以下为 codec 控制标记的 token id，决定生成的起止/思考流程：
        #   codec_eos_token_id  音频流结束标记
        #   codec_think_id      开启思考模式（模型先内部规划再生成）
        #   codec_nothink_id    关闭思考模式（直接生成）
        #   codec_think_bos_id  思考段起始标记
        #   codec_think_eos_id  思考段结束标记
        #   codec_pad_id        填充标记（对齐多码本时补位用）
        #   codec_bos_id        音频流起始标记
        self.codec_eos_token_id = codec_eos_token_id
        self.codec_think_id = codec_think_id
        # 语言字典：预置语言 id → 语言信息，控制合成语音的语言/方言
        self.codec_language_id = codec_language_id
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        # 说话人字典：预置音色 id → 说话人信息，用于选择内置音色
        self.spk_id = spk_id
        # 说话人是否为方言的标记字典
        self.spk_is_dialect = spk_is_dialect


class Qwen3TTSConfig(PretrainedConfig):
    # -----------------------------------------------------------------------------
    # 顶层配置：Qwen3-TTS 整体模型配置
    # -----------------------------------------------------------------------------
    # 用途：组合 talker（主生成模块）与 speaker_encoder（说话人编码器），
    #       并声明 tokenizer 类型、模型规模、模型类型与若干特殊 token id。
    #       它对应 Qwen3TTSForConditionalGeneration 这个端到端模型。
    # 关键参数：
    #   talker_config          主 talker 的配置（Qwen3TTSTalkerConfig）
    #   speaker_encoder_config 说话人编码器配置（Qwen3TTSSpeakerEncoderConfig）
    #   tokenizer_type         tokenizer 类型：25hz 或 12hz（决定 codec 帧率）
    #   tts_model_size         模型规模：0.6B 或 1.7B
    #   tts_model_type         模型类型：base / custom_voice / voice_design
    #   im_start/end_token_id  对话/图像块起止标记 id
    #   tts_pad/bos/eos_token_id  TTS 流的填充/起始/结束标记 id
    # -----------------------------------------------------------------------------
    """
    This is the configuration class to store the configuration of a [`Qwen3TTSForConditionalGeneration`]. 
    （中文：Qwen3-TTS 顶层配置类，组合 talker 与 speaker_encoder，并声明
     tokenizer 类型、模型规模、模型类型及特殊 token id。）
    """

    model_type = "qwen3_tts"
    sub_configs = {
        "talker_config": Qwen3TTSTalkerConfig,
        "speaker_encoder_config": Qwen3TTSSpeakerEncoderConfig,
    }

    def __init__(
        self,
        talker_config=None,
        speaker_encoder_config=None,
        tokenizer_type=None,
        tts_model_size=None,
        tts_model_type=None,
        im_start_token_id=151644,
        im_end_token_id=151645,
        tts_pad_token_id=151671,
        tts_bos_token_id=151672,
        tts_eos_token_id=151673,
        **kwargs,
    ):
        # 参数说明：
        #   talker_config            主 talker 配置（dict 或 Qwen3TTSTalkerConfig）
        #   speaker_encoder_config   说话人编码器配置（dict 或 Qwen3TTSSpeakerEncoderConfig）
        #   tokenizer_type           tokenizer 类型：'25hz' 或 '12hz'，决定 codec 帧率
        #                            （25hz 每秒 25 帧，12hz 每秒 12 帧；帧率影响时长建模粒度）
        #   tts_model_size           模型规模：'0.6B' 或 '1.7B'
        #   tts_model_type           模型类型：'base'(基础音色) / 'custom_voice'(声音克隆)
        #                            / 'voice_design'(声音设计)
        #   im_start_token_id        对话/图像块起始标记 id
        #   im_end_token_id          对话/图像块结束标记 id
        #   tts_pad_token_id         TTS 流填充标记 id（多码本对齐补位）
        #   tts_bos_token_id         TTS 流起始标记 id
        #   tts_eos_token_id         TTS 流结束标记 id
        super().__init__(**kwargs)

        # talker_config 为空时用默认配置实例化主 talker
        if talker_config is None:
            talker_config = {}
            logger.info("talker_config is None. Initializing talker model with default values")
        # speaker_encoder_config 为空时用默认配置实例化说话人编码器
        if speaker_encoder_config is None:
            speaker_encoder_config = {}
            logger.info("speaker_encoder_config is None. Initializing talker model with default values")

        # 用 talker_config 字典实例化主 talker 配置
        self.talker_config = Qwen3TTSTalkerConfig(**talker_config)
        # 用 speaker_encoder_config 字典实例化说话人编码器配置
        self.speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig(**speaker_encoder_config)

        # tokenizer 类型：决定 codec 帧率（25hz/12hz）
        self.tokenizer_type = tokenizer_type
        # 模型规模标识（0.6B / 1.7B）
        self.tts_model_size = tts_model_size
        # 模型类型标识（base / custom_voice / voice_design）
        self.tts_model_type = tts_model_type

        # 顶层特殊 token id：对话块起止 + TTS 流的填充/起始/结束标记
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id


__all__ = ["Qwen3TTSConfig", "Qwen3TTSTalkerConfig", "Qwen3TTSSpeakerEncoderConfig"]
