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
"""PyTorch Qwen3TTSTokenizerV2 model."""
# =============================================================================
# 文件总览：Qwen3-TTS 语音 tokenizer（12Hz 版，正式发布版 V2）
# -----------------------------------------------------------------------------
# 本文件是 Qwen3-TTS 系统中负责"语音 ↔ 离散编号"互转的核心模块。
# 整个 Qwen3-TTS 的思路是：把语音"分词"成一串离散编号（像文字 token 一样），
# 再交给大语言模型（LLM）去生成；最后把生成出的编号还原成可听的声音波形。
#
# 本文件定义的 Qwen3TTSTokenizerV2Model 就是这个"语音分词器"的 PyTorch 实现，
# 它包含两个核心部件：
#
#   1) encoder（编码器）：基于 Mimi 架构（卷积 + Transformer + 残差向量量化 RVQ）
#      把音频波形（24kHz 采样）压成 16 层离散编号 codes。
#      帧率 12.5Hz：每秒 12.5 个时间步，每步输出 16 层编号，
#      每层从一本含 2048 项的"码本"中挑一个最像的声音碎片编号。
#      下采样率 = 24000 / 12.5 = 1920 个采样点/步。
#
#   2) decoder（解码器）：把 16 层编号还原成波形
#      流程：RVQ 解码 → 预卷积 → 语义 Transformer → 上采样卷积块 → 声码器卷积
#      其中语义 Transformer 用滑动窗口注意力 + RoPE 旋转位置编码，
#      声码器部分用 SnakeBeta 激活函数（对音频更友好）+ 因果卷积，
#      支持流式因果生成。
#
# 数据流总结：
#   encode: 波形 (B, T_audio) ──Mimi──> codes (B, 16, T_frames)   T_frames≈T_audio/1920
#   decode: codes (B, 16, T_frames) ──Transformer+声码器──> 波形 (B, T_audio)
#
# 本文件在整体架构中的位置：
#   它是 Qwen3-TTS 把语音离散化的"前端"，输出的编号序列会喂给 LLM；
#   LLM 生成新编号序列后，再用本文件的 decoder 还原成语音。
# =============================================================================

import math
from dataclasses import dataclass
from typing import Callable, Optional, Union, List

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from transformers import MimiConfig, MimiModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, auto_docstring, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs

from .configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
    Qwen3TTSTokenizerV2DecoderConfig,
)

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV2EncoderOutput(ModelOutput):
    """编码器（encode）的输出：一段音频对应的离散编号 codes。

    每条音频会被压成 (codes_length, num_quantizers) 的编号矩阵，
    其中 num_quantizers=16 表示 16 层残差向量量化（RVQ）的编号。"""
    r"""
    audio_codes (`List[torch.LongTensor]`):
        Discret code embeddings computed using `model.encode`, each tensor has shape (codes_length_i, num_quantizers).
    """

    audio_codes: List[torch.LongTensor] = None


@dataclass
@auto_docstring
class Qwen3TTSTokenizerV2DecoderOutput(ModelOutput):
    """解码器（decode）的输出：把编号还原后的音频波形。

    每条编号序列会还原成一维的音频采样值序列。"""
    r"""
    audio_values (`List[torch.FloatTensor]`):
        Decoded audio values, obtained using the decoder part of Qwen3TTSTokenizerV1.
        Each tensor has shape (segment_length_i).
    """

    audio_values: List[torch.FloatTensor] = None


def rotate_half(x):
    """把输入向量后半部分旋转：(-x2, x1)，供 RoPE 旋转位置编码使用。

    RoPE（旋转位置编码）通过把 q/k 向量按位置旋转来注入位置信息，
    这里构造出"旋转另一半"的部分，配合 cos/sin 做旋转。"""
    x1 = x[..., : x.shape[-1] // 2]  # 前半段
    x2 = x[..., x.shape[-1] // 2 :]  # 后半段
    return torch.cat((-x2, x1), dim=-1)  # 拼成 (-后半, 前半)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """对 query 和 key 应用 RoPE 旋转位置编码。

    通俗理解：把"第几个位置"这个信息通过旋转"揉"进 q、k 向量里，
    让模型在算注意力时能感知到词/帧之间的相对距离。
    """
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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA（分组查询注意力）中把 KV 头复制 n_rep 份，对齐 Q 的头数。

    通俗理解：Q（查询）的头数多，K/V（键值）的头数少，为了能一一对应算注意力，
    就把每个 K/V 头复制 n_rep 份摊开，省内存又保效果。
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states  # 头数已经一致，无需复制
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
    """朴素的（非 flash）注意力前向实现，作为 fallback。

    步骤：Q·Kᵀ/缩放 → 加 mask → softmax → dropout → ·V。
    支持 GQA（先 repeat_kv）和滑动窗口（mask 已在外面准备好）。
    """
    key_states = repeat_kv(key, module.num_key_value_groups)  # K 头复制到与 Q 同数
    value_states = repeat_kv(value, module.num_key_value_groups)  # V 同上

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # 注意力打分 Q·Kᵀ * scale
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]  # 截到当前 K 长度
        attn_weights = attn_weights + causal_mask  # 加入因果/滑动窗口 mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)  # 归一化为概率
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)  # 训练时 dropout
    attn_output = torch.matmul(attn_weights, value_states)  # 用概率加权 V
    attn_output = attn_output.transpose(1, 2).contiguous()  # 调回 (B, seq, heads, dim) 布局

    return attn_output, attn_weights


@auto_docstring
class Qwen3TTSTokenizerV2DecoderPreTrainedModel(PreTrainedModel):
    """解码器（decoder）所有子模块的预训练基类。

    提供权重初始化、梯度检查点、flash attention / sdpa 支持等通用配置，
    decoder 的各部件（Transformer、声码器等）都继承自它。"""
    config: Qwen3TTSTokenizerV2DecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


class Qwen3TTSTokenizerV2CausalConvNet(nn.Module):
    """因果一维卷积（Causal Conv1d）。

    "因果"意味着：当前时刻的输出只依赖当前及过去的输入，不看未来，
    这对语音流式生成至关重要——生成第 t 帧时不能用第 t+1 帧的信息。
    实现方式是在左侧补 (kernel-1)*dilation 个零（只补左边，不补右边）。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1  # 感受野宽度（含 dilation 拓展）
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride  # 左侧因果 padding，保证只看过去

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        """算"额外右侧补零数"，使长度恰能被 stride 整除，避免下采样时丢帧。"""
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, hidden_state):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(hidden_state, (self.padding, extra_padding), mode="constant", value=0)  # 左补因果，右补整除
        return self.conv(hidden_state).contiguous()


class Qwen3TTSTokenizerV2CausalTransConvNet(nn.Module):
    """因果转置一维卷积（ConvTranspose1d），用于上采样（把序列变长）。

    解码器需要把低帧率的隐状态逐步还原成高采样率的波形，
    转置卷积按 stride 间隔插入并加权求和实现"放大"。右侧裁掉对齐 padding。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)

        pad = kernel_size - stride
        self.left_pad = 0
        self.right_pad = int(pad)  # 右侧要裁掉的长度

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        if self.right_pad > 0:
            hidden_state = hidden_state[..., : hidden_state.shape[-1] - self.right_pad]  # 裁掉右侧多余部分
        return hidden_state.contiguous()


class Qwen3TTSTokenizerV2ConvNeXtBlock(nn.Module):
    """ConvNeXt 风格的卷积残差块（现代卷积版 Transformer 块）。

    结构：深度可分离 7×1 卷积 → LayerNorm → 逐点升维(×4) → GELU → 逐点降维 → LayerScale → 残差相加。
    相比传统 ResNet 卷积块更接近 Transformer 的设计，效果好且省算力。"""

    def __init__(self, dim: int):
        super().__init__()
        # 深度卷积：每个通道单独卷，groups=dim，kernel=7（大感受野）
        self.dwconv = Qwen3TTSTokenizerV2CausalConvNet(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=1,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)  # 通道层归一化
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 逐点卷积升维 4 倍（等价于 1×1 卷积）
        self.act = nn.GELU()  # GELU 激活
        self.pwconv2 = nn.Linear(4 * dim, dim)  # 再降回原维
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))  # LayerScale：每通道一个极小可学习缩放系数，稳定训练

    def forward(self, hidden_states):
        input = hidden_states

        hidden_states = self.dwconv(hidden_states)  # 深度卷积
        hidden_states = hidden_states.permute(0, 2, 1)  # (B, C, T) -> (B, T, C) 以做逐点 Linear
        hidden_states = self.norm(hidden_states)  # 归一化
        hidden_states = self.pwconv1(hidden_states)  # 升维
        hidden_states = self.act(hidden_states)  # 激活
        hidden_states = self.pwconv2(hidden_states)  # 降维

        hidden_states = self.gamma * hidden_states  # LayerScale 缩放

        hidden_states = hidden_states.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

        hidden_states = input + hidden_states  # 残差连接

        return hidden_states


class Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(nn.Module):
    """解码器用的 RoPE 旋转位置编码（Rotary Position Embedding）。

    作用：给序列中每个位置算一组 cos/sin，供注意力层把位置"旋转"进 q/k。
    用不同频率的复指数，让相对距离天然蕴含在旋转中，外推性比传统位置编码好。"""
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings  # 缓存的最大序列长度
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]  # 选 RoPE 初始化函数（含动态/线性缩放等）

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)  # 基础频率表 + 注意力缩放
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 1/频率 表，决定旋转角速度
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        """根据位置 ids 算出每个位置的 cos/sin，返回 (cos, sin) 供 attention 使用。"""
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)  # pos × 1/freq = 角度
            emb = torch.cat((freqs, freqs), dim=-1)  # 拼成 2 倍宽度（cos 和 sin 用同一 freqs）
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3TTSTokenizerV2DecoderAttention(nn.Module):
    """解码器用的多头注意力（含 GQA 分组查询 + 滑动窗口 + RoPE）。

    - GQA：Q 头多，K/V 头少，省 KV cache 显存；
    - 滑动窗口：只看局部窗口内的历史，省算力，适合长序列流式生成；
    - RoPE：用上面的旋转编码注入位置。"""
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads  # 每个 KV 头摊给几个 Q 头
        self.scaling = self.head_dim**-0.5  # 注意力缩放因子 1/√d
        self.attention_dropout = config.attention_dropout
        self.is_causal = True  # 因果：不看未来

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )  # Q 投影
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )  # K 投影（头数更少）
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )  # V 投影（头数更少）
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )  # 输出投影
        self.q_norm = nn.Identity()  # 占位（这里不做 QK 归一化）
        self.k_norm = nn.Identity()
        self.sliding_window = config.sliding_window  # 滑动窗口大小：只看过去这么多步

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """输入 hidden_states (B, T, hidden)，输出注意力后的 (B, T, hidden) 和权重。"""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # (B, n_q, T, d)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # (B, n_kv, T, d)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # (B, n_kv, T, d)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # 注入 RoPE 位置

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)  # 累积 KV cache

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  # 优先用 flash/sdpa

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama  # 与 Llama 的区别：传滑动窗口
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # 合并多头
        attn_output = self.o_proj(attn_output)  # 输出投影回 hidden_size
        return attn_output, attn_weights


class Qwen3TTSTokenizerV2DecoderMlp(nn.Module):
    """解码器前馈网络（MLP），用 SwiGLU 风格：gate_proj·act × up_proj → down_proj。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size  # 升维后的中间维度
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # 门控分支
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # 值分支
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # 降回原维
        self.act_fn = ACT2FN[config.hidden_act]  # 通常为 SiLU

    def forward(self, x):
        # SwiGLU: down(act(gate(x)) * up(x))
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3TTSTokenizerV2DecoderRMSNorm(nn.Module):
    """RMSNorm：只用均方根做归一化（不抠均值），比 LayerNorm 更省计算。

    公式：x / sqrt(mean(x²) + eps) * weight。"""

    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3TTSTokenizerV2DecoderRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习缩放
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)  # 用 float32 算归一化保证稳定
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # 均方
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # 除以 RMS
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3TTSTokenizerV2DecoderLayerScale(nn.Module):
    """LayerScale：给每层输出逐通道乘一个可学习小系数，初始化接近 0。

    作用：训练初期让残差分支接近"恒等"，稳定深层网络的训练。"""
    """Layer scale from [Touvron et al 2021] (https://huggingface.co/papers/2103.17239).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.
    """

    def __init__(self, config):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale  # 初始小值（如 1e-5）
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, requires_grad=True))  # 每通道一个可学习系数

    def forward(self, x: torch.Tensor):
        return self.scale * x


class Qwen3TTSTokenizerV2DecoderTransformerLayer(GradientCheckpointingLayer):
    """解码器中单个 Transformer 层（残差式）：RMSNorm → 自注意力 → LayerScale+残差 → RMSNorm → MLP → LayerScale+残差。

    本层使用滑动窗口注意力（sliding_attention），只看局部历史窗口，适合长语音流式解码。"""

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSTokenizerV2DecoderAttention(config, layer_idx)  # 自注意力
        self.mlp = Qwen3TTSTokenizerV2DecoderMlp(config)  # 前馈
        self.input_layernorm = Qwen3TTSTokenizerV2DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)  # 进注意力前归一化
        self.post_attention_layernorm = Qwen3TTSTokenizerV2DecoderRMSNorm(config.hidden_size, config.rms_norm_eps)  # 进 MLP 前归一化
        self.self_attn_layer_scale = Qwen3TTSTokenizerV2DecoderLayerScale(config)  # 注意力残差的 LayerScale
        self.mlp_layer_scale = Qwen3TTSTokenizerV2DecoderLayerScale(config)  # MLP 残差的 LayerScale
        self.attention_type = "sliding_attention"  # 本层用滑动窗口注意力

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)  # 进注意力前归一化

        # Self Attention
        hidden_states, _ = self.self_attn(  # 自注意力（滑动窗口 + RoPE + GQA）
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)  # 残差 + LayerScale

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)  # 进 MLP 前归一化
        hidden_states = self.mlp(hidden_states)  # 前馈网络
        hidden_states = residual + self.mlp_layer_scale(hidden_states)  # 残差 + LayerScale

        return hidden_states


@auto_docstring
class Qwen3TTSTokenizerV2DecoderTransformerModel(Qwen3TTSTokenizerV2DecoderPreTrainedModel):
    """解码器的语义 Transformer 部分。

    作用：把 RVQ 解码出的连续声学特征做一次"语义级"建模（建模长程依赖、上下文），
    再交给后面的卷积声码器去合成波形。这里用滑动窗口因果 Transformer，支持 KV cache 流式。"""
    _can_record_outputs = {
        "hidden_states": Qwen3TTSTokenizerV2DecoderTransformerLayer,
        "attentions": Qwen3TTSTokenizerV2DecoderAttention,
    }

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3TTSTokenizerV2DecoderTransformerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )  # 堆叠 N 层 Transformer
        self.norm = Qwen3TTSTokenizerV2DecoderRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最后一层后归一化
        self.rotary_emb = Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(config=config)  # RoPE 位置编码
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types  # 是否含滑动窗口层
        self.window_size = config.sliding_window  # 滑动窗口大小

        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)  # 把 latent 投影到 Transformer 隐维
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)  # 再投回 latent 维

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """前向：输入连续嵌入 (B, T, latent)，输出隐状态 (B, T, latent) + KV cache。

        本 Transformer 接收来自 RVQ 解码+预卷积的连续特征，做语义建模后投影回 latent 维。"""
        if input_ids is not None:
            raise ValueError("input_ids is not expected")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        inputs_embeds = self.input_proj(inputs_embeds)  # latent_dim -> hidden_size

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)  # 新建 KV cache

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(  # 当前 batch 在序列中的绝对位置
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)  # 默认位置 = cache_position

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),  # 全因果 mask（不看未来）
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)  # 滑动窗口 mask：只看局部窗口

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)  # 共享同一组 RoPE cos/sin 给所有层

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(  # 逐层前向
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],  # 按层类型选 mask
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)  # 末层归一化
        hidden_states = self.output_proj(hidden_states)  # hidden_size -> latent_dim
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class SnakeBeta(nn.Module):
    """SnakeBeta 激活函数：专为音频/周期信号设计的激活，比 ReLU 平滑。

    公式：x + 1/b * sin²(x·a)，其中 a、b 是可学习参数（控制频率与幅度）。
    - a（alpha）控制周期分量的频率；
    - b（beta）控制周期分量的幅度。
    因为它带周期性（sin），很适合建模语音这种准周期波形，能减少高频失真。
    """
    """
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
        self.alpha = Parameter(torch.zeros(in_features) * alpha)  # 控制频率，初始化为 0
        self.beta = Parameter(torch.zeros(in_features) * alpha)  # 控制幅度

        self.no_div_by_zero = 0.000000001  # 防止除零

    def forward(self, hidden_states):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]  # 对齐到 (B, C, T)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)  # 用 exp 保证非负，等价于学习 log(a)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )  # x + (1/b) * sin²(x·a)

        return hidden_states


class Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(nn.Module):
    """声码器中的残差单元：SnakeBeta → 7×1 因果卷积（带 dilation）→ SnakeBeta → 1×1 卷积，再残差相加。

    用不同 dilation（1/3/9）堆叠多个，能扩大感受野又不丢细节，适合建模多尺度时频结构。"""

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()

        self.act1 = SnakeBeta(dim)
        self.conv1 = Qwen3TTSTokenizerV2CausalConvNet(dim, dim, kernel_size=7, dilation=dilation)  # 大核 + dilation 扩感受野
        self.act2 = SnakeBeta(dim)
        self.conv2 = Qwen3TTSTokenizerV2CausalConvNet(dim, dim, kernel_size=1)  # 1×1 做通道混合

    def forward(self, hidden_state):
        residual = hidden_state

        hidden_state = self.act1(hidden_state)  # SnakeBeta 激活
        hidden_state = self.conv1(hidden_state)  # 因果膨胀卷积
        hidden_state = self.act2(hidden_state)  # 再激活
        hidden_state = self.conv2(hidden_state)  # 1×1 卷积
        return hidden_state + residual  # 残差


class Qwen3TTSTokenizerV2DecoderDecoderBlock(Qwen3TTSTokenizerV2DecoderPreTrainedModel):
    """声码器的一个上采样块：SnakeBeta → 转置卷积上采样（×upsample_rate）→ 3 个不同 dilation 的残差单元。

    每过一块，通道数减半、时间长度翻 upsample_rate 倍，把低帧率特征逐步恢复到高采样率。"""

    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig, layer_idx):
        super().__init__(config)
        in_dim = config.decoder_dim // 2**layer_idx  # 通道数随层减半
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]  # 本层上采样倍率

        block = [
            SnakeBeta(in_dim),
            Qwen3TTSTokenizerV2CausalTransConvNet(in_dim, out_dim, 2 * upsample_rate, upsample_rate),  # 转置卷积上采样
        ]

        for dilation in (1, 3, 9):  # 三种膨胀率，扩大感受野层次
            block.append(Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(out_dim, dilation))

        self.block = nn.ModuleList(block)

    def forward(self, hidden):
        for block in self.block:
            hidden = block(hidden)  # 依次跑激活→上采样→3 个残差单元
        return hidden


class EuclideanCodebook(nn.Module):
    """欧氏码本：存 codebook_size 个"标准声音碎片"向量（维度 dim）。

    量化时把连续向量近似成码本里欧氏距离最近的那一项的编号；
    解码（decode）时只需按编号从码本取回对应向量。
    用 embedding_sum/cluster_usage 的指数移动平均来更新码本（训练时）。"""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size  # 码本大小，如 2048
        self.epsilon = epsilon

        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))  # 每个码本项的使用计数（EMA）
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))  # 各项向量的累加和（EMA）

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """按编号 codes 从码本取回向量。

        输入 codes 形状任意（整数编号），输出为对应码本向量，最后一维 = dim。"""
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]  # 由 EMA 求平均得到码本向量
        quantized = F.embedding(codes, embedding)  # 查表：按编号取向量
        return quantized


class VectorQuantization(nn.Module):
    """单本向量量化（VQ）：把连续向量近似成码本里最近的一项。

    这里只实现 decode 方向（编号 → 向量），量化训练逻辑在外部（encoder 的 Mimi 里）。"""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim  # 维度不一致时需要 1×1 投影对齐

        self.project_out = (
            nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        )  # 把码本维度投回模型维度
        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(
            dim=codebook_dim,
            codebook_size=codebook_size,
            epsilon=epsilon
        )
        self.codebook_size = codebook_size

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """编号 → 量化向量，并把通道维转置到第二维 (B, dim, T)。"""
        quantized = self._codebook.decode(codes)  # 查码本
        quantized = self.project_out(quantized)  # 投影回模型维度
        quantized = quantized.transpose(1, 2)  # (B, T, dim) -> (B, dim, T)
        return quantized


class ResidualVectorQuantization(nn.Module):
    """残差向量量化（RVQ）：堆叠 num_quantizers 本码本，层层逼近。

    第 1 本量化原信号 → 算残差；第 2 本量化残差 → 再算残差……
    每本只记"上一本没拟合的误差"，16 本叠起来精度很高，
    而每本只用一个编号，所以总编码是 (16, T) 的编号序列。"""

    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]  # num_quantizers 本 VQ
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """逐层解码并累加：每本 decode 出的向量相加，得到最终量化结果。

        codes 形状 (num_quantizers, T)（外层已 transpose 成 (K, T)）。"""
        quantized = torch.zeros([1], device=codes.device)[0]  # 初始为 0 向量
        for idx, layer_codes in enumerate(codes):  # 逐层
            layer = self.layers[idx]
            assert isinstance(layer, VectorQuantization)
            quantized = quantized + layer.decode(layer_codes)  # 累加本层的贡献
        return quantized


class ResidualVectorQuantizer(nn.Module):
    """RVQ 的封装：在前后加可选的 1×1 卷积做维度投影。

    input_proj 把输入维 input_dimension 映到量化维 dimension；
    output_proj 把量化维映回 output_dimension。
    训练相关参数（n_q、q_dropout、no_quantization_rate、decay 等）这里保留接口。"""

    def __init__(
        self,
        dimension: int = 128,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        n_q: int = 8,  # 残差量化的本数（层数）
        q_dropout: bool = False,  # 训练时随机丢弃若干层 RVQ，增强鲁棒性
        no_quantization_rate: float = 0.0,
        bins: int = 1024,  # 码本大小
        decay: float = 0.99,  # EMA 衰减系数
        force_projection: bool = False,  # 即使维度相同也强制加投影
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        self.decay = decay
        self.input_proj: torch.nn.Module
        self.output_proj: torch.nn.Module
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(
                self.input_dimension, self.dimension, 1, bias=False
            )  # 1×1 卷积做输入投影
        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(
                self.dimension, self.output_dimension, 1, bias=False
            )  # 1×1 卷积做输出投影
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """把编号 codes 解码成连续向量。

        codes: (B, K, T) → transpose 成 (K, B, T) → RVQ 累加 → output_proj 投影。"""
        codes = codes.transpose(0, 1)  # (B, K, T) -> (K, B, T)
        quantized = self.vq.decode(codes)  # 逐层累加
        quantized = self.output_proj(quantized)  # 投影回输出维
        return quantized


class SplitResidualVectorQuantizer(nn.Module):
    """把 RVQ 分成两段：前 n_q_semantic 本（语义层）+ 剩余 n_q_acoustic 本（声学层）。

    语义层（通常第 1 本）主要承载语言内容信息，会单独喂给/对齐 LLM；
    声学层承载说话人、音色、细节等。两段各自有投影，解码时分别 decode 再相加。"""
    """Residual Vector Quantizer with separate projections for the first quantizer and the rest.

    Args:
        n_q (int): Number of residual vector quantizers used.
        n_semantic_q (int): Number of residual vector quantizers used for the semantic quantizer.
        **kwargs: Arguments to the constructor of `ResidualVectorQuantizer` that are shared between both.
    """

    def __init__(
        self,
        *,
        n_q: int = 8,
        n_q_semantic: int = 1,  # 前 n_q_semantic 本作为语义层
        **kwargs,
    ):
        super().__init__()
        assert n_q > n_q_semantic, (
            f"Number of quantizers {n_q} must be larger "
            f"than the number of semantic quantizers {n_q_semantic}."
        )
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic  # 剩余作为声学层
        q_dropout = kwargs.pop("q_dropout", False)
        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )  # 语义层 RVQ
        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            force_projection=True,
            q_dropout=q_dropout,
            **kwargs,
        )  # 声学层 RVQ

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """解码编号 codes (B, K, T)：语义层 + 声学层分别 decode 后相加。"""
        # codes is [B, K, T], with T frames, K nb of codebooks.
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])  # 语义层
        if codes.shape[1] > self.n_q_semantic:
            quantized += self.rvq_rest.decode(codes[:, self.n_q_semantic :])  # 声学层
        return quantized


class Qwen3TTSTokenizerV2Decoder(Qwen3TTSTokenizerV2DecoderPreTrainedModel):
    """语音 tokenizer 的解码器（decode）：把 16 层离散编号还原成波形。

    完整流程：codes (B, K, T) → ① RVQ 解码成连续向量 → ② 预卷积 → ③ 语义 Transformer
    → ④ 上采样块（转置卷积 + ConvNeXt）→ ⑤ 声码器卷积块 → 波形 (B, 1, T_audio)。
    全程因果卷积 + 滑动窗口注意力，支持流式解码。"""
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfig):
        super().__init__(config)
        # 总上采样倍率：upsample_rates 与 upsampling_ratios 的乘积，1 帧 → total_upsample 个采样点
        self.total_upsample = np.prod(config.upsample_rates + config.upsampling_ratios)
        self.pre_transformer = Qwen3TTSTokenizerV2DecoderTransformerModel._from_config(config)  # 语义 Transformer
        
        self.quantizer = SplitResidualVectorQuantizer(  # RVQ 解码（语义层 1 本 + 声学层 K-1 本）
            dimension=config.codebook_dim // 2,
            n_q=config.num_quantizers,
            n_q_semantic=1,
            bins=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )

        self.pre_conv = Qwen3TTSTokenizerV2CausalConvNet(  # RVQ 解码后的 3×1 因果卷积，调维到 latent_dim
            config.codebook_dim,
            config.latent_dim,
            kernel_size=3,
        )

        upsample = []
        for factor in config.upsampling_ratios:  # 上采样阶段：每个 factor 做一次转置卷积+ConvNeXt 残差
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3TTSTokenizerV2CausalTransConvNet(config.latent_dim, config.latent_dim, factor, factor),
                        Qwen3TTSTokenizerV2ConvNeXtBlock(config.latent_dim),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        # 声码器：先 7×1 卷积升到 decoder_dim，再堆 len(upsample_rates) 个解码块逐级上采样
        decoder = [Qwen3TTSTokenizerV2CausalConvNet(config.latent_dim, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3TTSTokenizerV2DecoderDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)  # 最后通道数
        decoder += [
            SnakeBeta(output_dim),  # 末尾 SnakeBeta 激活
            Qwen3TTSTokenizerV2CausalConvNet(output_dim, 1, 7),  # 7×1 卷积降到单通道波形
        ]
        self.decoder = nn.ModuleList(decoder)

        self.post_init()

    def forward(self, codes):
        """单段解码：codes (B, K, T) → 波形 (B, 1, T_audio)。

        K 必须等于 num_quantizers（16），T 是帧数，T_audio ≈ T × total_upsample。"""
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}")

        hidden = self.quantizer.decode(codes)  # ① RVQ 解码：(B, codebook_dim, T)
        hidden = self.pre_conv(hidden).transpose(1, 2)  # ② 预卷积后 (B, T, latent)

        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state  # ③ 语义 Transformer：(B, T, latent)
        hidden = hidden.permute(0, 2, 1)  # (B, T, latent) -> (B, latent, T)
        for blocks in self.upsample:  # ④ 上采样：每个 factor 翻倍长度
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:  # ⑤ 声码器卷积：逐步上采样到波形采样率并降到单通道
            wav = block(wav)
        return wav.clamp(min=-1, max=1)  # 波形截断到 [-1, 1]，标准 PCM 范围

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        """分块流式解码：把长 codes 切成 chunk_size 帧一段，每段带 left_context_size 帧历史上下文，
        分别解码后拼接，避免一次性算整段、也便于流式输出。

        拼接时裁掉每段开头那段属于"上下文"的波形（context_size × total_upsample 个采样点）。"""
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index  # 首段没有完整上下文
            codes_chunk = codes[..., start_index - context_size : end_index]  # 带左侧上下文的切片
            wav_chunk = self(codes_chunk)  # 解码这段
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])  # 裁掉上下文对应波形，只留本段
            start_index = end_index
        return torch.cat(wavs, dim=-1)  # 拼接所有段


class Qwen3TTSTokenizerV2Encoder(MimiModel):
    """语音 tokenizer 的编码器（encode）：基于 Mimi 架构，把波形压成 16 层离散编号。

    Mimi = 卷积下采样 + 语义/声学 Transformer + 残差向量量化（RVQ）。
    输入 24kHz 波形，输出帧率 12.5Hz 的 codes（每帧 16 层编号，每层从 2048 项码本选一个）。
    本类直接复用 transformers 的 MimiModel，并把 decoder 相关字段置空（decoder 由本文件的 V2Decoder 单独实现）。"""
    def __init__(self, config: MimiConfig):
        super().__init__(config)
        self.config = config

        self.upsample = None  # encoder 不需要上采样，置空
        self.decoder_transformer = None  # decoder 的 transformer 由 V2Decoder 单独管理
        self.decoder = None  # decoder 由 V2Decoder 单独管理

        self.post_init()


@auto_docstring
class Qwen3TTSTokenizerV2PreTrainedModel(PreTrainedModel):
    """顶层模型 Qwen3TTSTokenizerV2Model 的预训练基类。"""
    config: Qwen3TTSTokenizerV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = False
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    The Qwen3TTSTokenizerV2 model.
    """
)
class Qwen3TTSTokenizerV2Model(Qwen3TTSTokenizerV2PreTrainedModel):
    """语音 tokenizer 顶层模型：组合 encoder（Mimi）和 decoder（V2），对外暴露 encode/decode。

    - encode：波形 → 16 层离散编号（喂给 LLM 生成）
    - decode：编号 → 波形（把 LLM 生成的编号还原成声音）
    """
    def __init__(self, config: Qwen3TTSTokenizerV2Config):
        super().__init__(config)
        self.config = config

        self.encoder_valid_num_quantizers = config.encoder_valid_num_quantizers  # encode 时实际取前几层编号

        self.input_sample_rate = config.input_sample_rate  # 输入音频采样率（24000Hz）
        self.output_sample_rate = config.output_sample_rate  # 输出波形采样率（24000Hz）

        self.decode_upsample_rate = config.decode_upsample_rate  # decode：1 帧 → 多少采样点（≈1920）
        self.encode_downsample_rate = config.encode_downsample_rate  # encode：多少采样点 → 1 帧（≈1920）

        self.encoder = Qwen3TTSTokenizerV2Encoder._from_config(self.config.encoder_config)  # Mimi 编码器
        self.decoder = Qwen3TTSTokenizerV2Decoder._from_config(self.config.decoder_config)  # V2 解码器

        self.post_init()
    
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
    
    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, Optional[torch.Tensor]], Qwen3TTSTokenizerV2EncoderOutput]:
        """编码：波形 → 离散编号 codes。

        输入 input_values (B, T_audio)（24kHz 波形）+ padding_mask (B, T_audio)（标记非 pad 的有效点），
        输出每条音频的 codes，形状 (codes_length, num_quantizers)，codes_length ≈ 有效采样点数 / 1920。
        """
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

        # 调 Mimi encoder：补一维通道后送入，得到 audio_codes (B, K_total, T)
        encoded_frames = self.encoder.encode(input_values=input_values.unsqueeze(1),
                                             return_dict=True)
        # 只取前 encoder_valid_num_quantizers 层（语义需要的那部分）
        audio_codes = encoded_frames.audio_codes[:, :self.encoder_valid_num_quantizers]
        # 按 padding_mask 算每条音频的有效帧数，裁掉尾部 padding 帧；并把层维和时间维 transpose 成 (T, K)
        audio_codes = [code[..., :-(-mask.sum() // self.encode_downsample_rate)].transpose(0, 1) for code, mask in zip(audio_codes, padding_mask)]

        if not return_dict:
            return (
                audio_codes,
            )

        return Qwen3TTSTokenizerV2EncoderOutput(audio_codes)

    def decode(
        self,
        audio_codes: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], Qwen3TTSTokenizerV2DecoderOutput]:
        """解码：离散编号 → 波形。

        输入 audio_codes (B, T, K)（编号，pad 处用 -1 标记），
        输出每条音频的波形，长度 = (有效帧数) × decode_upsample_rate。
        注意：输出可能比输入对应长度略长，尾部多余可裁掉。"""
        """
        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, codes_length, num_quantizers)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            return_dict (`bool`, *optional*):
                Whether or to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # 用第 0 层编号 > -1 的位置算每条音频的有效帧数，再乘上采样率得到目标波形长度
        audio_lengths = (audio_codes[..., 0] > -1).sum(1) * self.decode_upsample_rate

        audio_codes = torch.clamp(audio_codes, min=0)  # 把 pad 处的 -1 截成 0，避免查码本越界
        # 调 decoder 分块流式解码；transpose 把 (B, T, K) -> (B, K, T) 以匹配 decoder 输入
        audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(1)

        audio_values = [a[:l] for a, l in zip(audio_values, audio_lengths)]  # 按目标长度裁掉尾部多余

        if not return_dict:
            return (
                audio_values,
            )

        return Qwen3TTSTokenizerV2DecoderOutput(audio_values)


__all__ = ["Qwen3TTSTokenizerV2Model", "Qwen3TTSTokenizerV2PreTrainedModel"]
