# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
本文件实现了基于 Whisper（OpenAI 开源的语音识别模型，这里只借用它的编码器部分提取音频特征）的音频编码器。

它在 Qwen3-TTS 的 25Hz tokenizer（分词器/标记器：把音频切成模型能处理的最小单元）流程中处于"特征提取"阶段：

    原始波形 (16kHz)
        │
        ▼  log_mel_spectrogram
    mel 频谱图 [n_mels, T_mel]     ← 人耳感知的时频图
        │
        ▼  conv1 (stride=1) + conv2 (stride=2)
    [n_state, T_mel/2]             ← 时间维减半，100 帧/秒 → 50 帧/秒
        │
        ▼  加正弦位置编码 + N 层 Transformer 编码器
    [T_mel/2, n_state]             ← 自注意力建模全局时序依赖
        │
        ▼  AvgPool1d(stride=2)
    [T_mel/4, n_state]             ← 时间维再减半，50 帧/秒 → 25 帧/秒（这就是"25Hz"的由来）
        │
        ▼  LayerNorm + Linear 投影
    [T_mel/4, output_dim=512]    ← 连续 latent（隐变量：模型内部学出的压缩特征，浮点数向量）
        │
        ▼  在每条音频首尾各插入一个 BOS/EOS 特殊 token
    输出给后续 VQ（Vector Quantization，矢量量化：把连续向量就近匹配到有限个"码本"样板上，变成整数编号）做离散化。

关键点：本文件输出的还是**连续 latent 向量**，不是离散 token；离散化由同目录下的 speech_vq.py 完成。
"""
import os
import math
import torch
import operator

import numpy as np
import torch.nn.functional as F

from functools import lru_cache
from typing import Optional, Union, List
from torch import nn, Tensor
from itertools import accumulate

# 可选依赖：flash-attn 是一种省显存、跑得快的注意力算子（把 Q/K/V 分块算，不落地完整注意力矩阵）。
# 装不上就回退到手写 PyTorch 版注意力，功能一样只是慢、占显存多。
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func
except ImportError:
    try:
        # 兼容老版本 flash-attn 的函数名
        from flash_attn.flash_attn_interface import flash_attn_unpadded_func as flash_attn_varlen_func
    except ImportError:
        print("\n********\nWarning: flash-attn is not installed. Will only run the manual PyTorch version. Please install flash-attn for faster inference.\n********\n ")
        flash_attn_varlen_func = None


# STFT（Short-Time Fourier Transform，短时傅里叶变换）窗口长度：400 个采样点。
# 16kHz 采样率下，400/16000 = 25ms 一帧窗。
N_FFT = 400
# 相邻两帧 STFT 窗之间跳过的采样点数：160，即 160/16000 = 10ms 一帧。
# 所以 mel 频谱的帧率 = 16000/160 = 100 帧/秒。
HOP_LENGTH = 160


@lru_cache(maxsize=None)  # lru_cache：缓存函数返回值，相同参数直接复用，避免反复读磁盘
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    # mel 滤波器组：一组三角形窗函数，把线性频率轴（0~8kHz）压缩成人耳更敏感的 mel 刻度。
    # 人耳对低频敏感、高频迟钝，所以 mel 刻度在低频段分得细、高频段分得粗。
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    # 滤波器矩阵预存在 assets/mel_filters.npz 里（80 维和 128 维两套），避免运行时依赖 librosa 库
    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        # 取出对应 n_mels 的矩阵，形状 [n_mels, N_FFT/2+1]，搬到目标设备（CPU/GPU）
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    # 把一维波形变成 log-mel 频谱图：输出形状 (n_mels, 帧数)。
    # 这是 Whisper 系列模型统一的音频前端——所有 Whisper 衍生模型都吃这个格式的输入。
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)  # 非 tensor（如 numpy 数组）先转成 tensor

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))  # 在右侧补 padding 个零采样点，让帧长对齐
    # 汉宁窗：形状像钟形，中间高两边低，用来减小 STFT 帧边界处的频谱泄漏
    window = torch.hann_window(N_FFT).to(audio.device)
    # STFT：把波形切成一段段短窗，每段做傅里叶变换。输出复数谱，形状 [N_FFT/2+1, 帧数]
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    # 取幅度（复数的模）并平方 → 功率谱。[..., :-1] 去掉最后一个对称频率 bin。
    # 形状从 [N_FFT/2+1, 帧数] 变成 [N_FFT/2, 帧数]
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)  # 取 mel 滤波器组，形状 [n_mels, N_FFT/2]
    # 矩阵相乘：功率谱投影到 mel 刻度。形状 [n_mels, N_FFT/2] @ [N_FFT/2, 帧数] = [n_mels, 帧数]
    mel_spec = filters @ magnitudes

    # 取对数（先 clamp 到 1e-10 以上防止 log(0) = -inf）。对数压缩把人耳感知的响度线性化
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # 动态范围压缩：比最大值低 8 个 dB 的部分截断，避免极小值噪声干扰训练
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    # 归一化到大致 [-1, 1] 附近，匹配 Whisper 的预处理约定
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def get_T_after_cnn(L_in, dilation=1):
    # 工具函数：给定输入帧数 L_in，计算两层卷积（conv1: kernel=3/stride=1/pad=1，conv2: kernel=3/stride=2/pad=1）之后的输出帧数。
    # 用途：提前知道卷积后时间维长度，用于构造 cu_seqlens。
    # eval("[(1,3,1)] + [(1,3,2)] ") 等价于 [(1,3,1), (1,3,2)]：(padding, kernel_size, stride)
    for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
        # PyTorch Conv1d 输出长度公式：L_out = floor((L_in + 2*pad - dilation*(k-1) - 1) / stride) + 1
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out


def get_mel_audio(audio, padding=False, audio_vq_ds_rate = 1, n_mels = 128):
    # 对外封装的"算 mel 频谱"入口。
    audio_len = len(audio)
    if padding:
        # 总下采样率 = hop(160 采样点/帧) × 卷积 stride(2) × VQ 下采样率。
        # padding=True 时把音频右侧补零到总下采样倍数的整数倍，避免最后一帧残缺。
        reduction = 160 * 2 * audio_vq_ds_rate
        audio_pad = math.ceil(audio_len / reduction) * reduction - audio_len  # 需要补的零采样点数
        mel = log_mel_spectrogram(audio, n_mels=n_mels, padding=audio_pad)
    else:
        mel = log_mel_spectrogram(audio, n_mels=n_mels)  # [F, T]：F=n_mels 频段维，T=时间帧维
    return mel


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    # 经典 Transformer 正弦位置编码（Vaswani et al. 2017）。
    # Transformer 本身没有位置概念，需要额外给每个时间步加一个"位置向量"，让模型知道谁在前谁在后。
    # 用不同频率的 sin/cos 拼接：高频位置区分相邻步，低频位置区分远距离步。
    assert channels % 2 == 0
    # 计算半通道数个不同频率的倒数时间尺度
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    # 位置矩阵 [length, channels//2]：每个位置 × 每个频率
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    # 前半通道放 sin，后半通道放 cos，拼成 [length, channels]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class Conv1d(nn.Conv1d):
    # 对 nn.Conv1d 的薄封装：前向时把权重/偏置 cast 到输入 x 的 dtype。
    # 原因：模型权重常以 FP32 存储，但推理时输入是 FP16/BF16，直接算会报类型不匹配。
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class ConvTranspose1d(nn.ConvTranspose1d):
    # 同上，转置卷积（上采样用）版本的 dtype 对齐封装。本编码器里没用到，是为解码器预留的。
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class Linear(nn.Linear):
    # 同上，全连接层版本的 dtype 对齐封装
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype) )


class MultiHeadAttention(nn.Module):
    """
    多头自注意力（Multi-Head Self-Attention）：让序列中每个位置都能"看到"其他所有位置，
    自动学习谁和谁相关。比如一个音节能关注前面的元音来推断它属于哪个音节。

    "多头"=把特征维切成 n_head 份，每份独立算注意力，最后拼回来，让模型从不同子空间理解关系。
    """
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        # Q = Query（查询）："我在找什么"
        self.query = Linear(n_state, n_state)
        # K = Key（键）："我能提供什么"，无偏置
        self.key = Linear(n_state, n_state, bias=False)
        # V = Value（值）："我的实际内容"
        self.value = Linear(n_state, n_state)
        # 注意力输出投影
        self.out = Linear(n_state, n_state)

        self.use_flash_attention = True  # 默认尝试用 flash-attn，运行时发现不满足条件会自动关掉

    def forward(
        self,
        x: Tensor,
        # cu_seqlens：cumulative sequence lengths，累积序列长度边界。
        # 多条不等长音频被"打包"(packed) 成一个长张量 [总帧数, n_state]，
        # cu_seqlens = [0, len1, len1+len2, ...] 标出每条音频在长张量里的起止位置。
        # 这样就不用 padding 到统一长度再算注意力（省显存），但注意力只能在同一条音频内部做。
        cu_seqlens = None,
    ):
        # Q/K/V 都是输入 x 的线性投影，形状 [总帧数, n_state]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if self.use_flash_attention:
            if flash_attn_varlen_func is None:
                # 没装 flash-attn：走手写实现
                x = self.qkv_attention_manual(q, k, v, cu_seqlens=cu_seqlens)
            else:
                # flash-attn 只支持 FP16/BF16；若当前不是半精度则退回手写实现并永久关闭
                if q.dtype not in [torch.float16, torch.bfloat16]:
                    x = self.qkv_attention_manual(q, k, v, cu_seqlens=cu_seqlens)
                    self.use_flash_attention = False
                else:
                    x = self.qkv_flash_attention(q, k, v, cu_seqlens=cu_seqlens)
        else:
            x = self.qkv_attention_manual(q, k, v, cu_seqlens=cu_seqlens)

        output = self.out(x)
        return output

    def qkv_flash_attention(
        self, q: Tensor, k: Tensor, v: Tensor, cu_seqlens=None
    ):
        # 使用 flash-attn 的 varlen（variable-length，变长）接口计算打包序列注意力。
        # flash-attn 的核心优化：不落地 [n_ctx, n_ctx] 的完整注意力矩阵（这个矩阵常占显存大头），
        # 而是分块算 Q·K^T，逐块和 softmax、V 相乘。
        n_ctx, n_state = q.shape
        # scale = (n_state // self.n_head) ** -0.25
        # 把 [n_ctx, n_state] 拆成 [n_ctx, n_head, head_dim]：每个头独立一份
        q = q.view(n_ctx, self.n_head, -1)# (batch_size, seqlen, nheads, headdim)
        k = k.view(n_ctx, self.n_head, -1)
        v = v.view(n_ctx, self.n_head, -1)

        # batch 内最长一条序列长度，flash-attn 需要这个来分配内部缓冲
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        # flash_attn_varlen_func 接受"打包"形状 (总长度, nheads, headdim) + 累积边界 cu_seqlens
        # 它内部会按 cu_seqlens 把每条音频切开，各自算各自的注意力，互不干扰
        x = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0
        )
        # 拼回 [n_ctx, n_state]
        x = x.reshape(n_ctx, n_state)
        return x

    def qkv_attention_manual(
        self, q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor
    ):
        # 手写版多头注意力（flash-attn 不可用时的回退方案）。
        # 做法：把打包的不等长序列"拆包"成 padded batch（补零到统一长度），
        # 用标准矩阵乘算注意力，再"打包"回去。
        n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head  # 每个头的特征维
        # 注意力分数缩放：除以 sqrt(head_dim)，防止点积值过大使 softmax 进入饱和区（梯度消失）
        scale = head_dim ** -0.5

        # 拆头：[n_ctx, n_state] → [n_ctx, n_head, head_dim]
        q = q.view(n_ctx, self.n_head, head_dim)
        k = k.view(n_ctx, self.n_head, head_dim)
        v = v.view(n_ctx, self.n_head, head_dim)

        # 从 cu_seqlens 反推出每条序列的真实长度
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()  # 每条序列真实长度
        batch_size = len(seqlens)
        max_seqlen = max(seqlens)

        # 分配 padded 张量 [batch, max_seqlen, n_head, head_dim]，把每条序列按真实长度填进去
        q_padded = torch.zeros(batch_size, max_seqlen, self.n_head, head_dim, dtype=q.dtype, device=q.device)
        k_padded = torch.zeros_like(q_padded)
        v_padded = torch.zeros_like(q_padded)

        for i in range(batch_size):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i+1]
            seq_len = seqlens[i]
            q_padded[i, :seq_len] = q[start_idx:end_idx]
            k_padded[i, :seq_len] = k[start_idx:end_idx]
            v_padded[i, :seq_len] = v[start_idx:end_idx]

        # 转到 [batch, n_head, max_seqlen, head_dim]，以便批量矩阵乘
        # 注意力矩阵形状是 [batch, n_head, max_seqlen, max_seqlen]
        q_padded = q_padded.transpose(1, 2)
        k_padded = k_padded.transpose(1, 2)
        v_padded = v_padded.transpose(1, 2)

        # 构造注意力掩码：真实长度之外的位置屏蔽掉。
        # attn_mask[i, j] = True 表示第 i 条序列的第 j 个位置有效
        attn_mask = torch.arange(max_seqlen, device=q.device)[None, :] < torch.tensor(seqlens, device=q.device)[:, None]
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, max_seqlen] 广播用

        # 无效位置填一个极小值（不是 0，因为 0 经 softmax 后会变成概率 1/batch_size，会污染结果）
        attn_mask = attn_mask.masked_fill(attn_mask == 0, -torch.finfo(q.dtype).max)

        # 标准注意力公式：softmax(Q·K^T / sqrt(d)) · V
        attn_scores = torch.matmul(q_padded, k_padded.transpose(-2, -1)) * scale  # [batch, n_head, max_seqlen, max_seqlen]
        attn_scores = attn_scores + attn_mask  # 加上屏蔽，无效位置 softmax 后近似为 0
        attn_weights = F.softmax(attn_scores, dim=-1)  # 概率分布

        context = torch.matmul(attn_weights, v_padded)  # [batch, n_head, max_seqlen, head_dim]

        # 拼回头维：[batch, max_seqlen, n_state]
        context = context.transpose(1, 2).contiguous().view(batch_size, max_seqlen, n_state)

        # 把 padded 结果按真实长度切回打包形式（去掉补的零），沿 dim=0 拼成长张量
        output_packed = torch.cat([context[i, :seqlens[i]] for i in range(batch_size)], dim=0)

        # 形状校验：打包后总帧数必须等于输入总帧数
        assert output_packed.shape == (n_ctx, n_state)

        return output_packed


class ResidualAttentionBlock(nn.Module):
    """
    一个 Transformer 编码器块（Pre-LN 结构）：
        x = x + Attention(LayerNorm(x))     ← 注意力残差子层
        x = x + MLP(LayerNorm(x))           ← 前馈残差子层

    Pre-LN = LayerNorm 在注意力/MLP 之前做（而不是之后），训练更稳定。
    残差连接（x + ...）让梯度能直接流过深层网络，缓解梯度消失。
    """
    def __init__(self, n_state: int, n_head: int,
                 enable_mp: bool = False, sequence_parallel: bool = False):
        super().__init__()
        n_mlp = n_state * 4  # 前馈隐层维度通常为模型维度的 4 倍（Transformer 标准设计）
        self.attn_ln = nn.LayerNorm(n_state)   # 注意力前的 LayerNorm
        self.mlp_ln = nn.LayerNorm(n_state)    # MLP 前的 LayerNorm

        self.attn = MultiHeadAttention(n_state, n_head)
        # MLP = Linear(GELU 激活)Linear：两层全连接，GELU 是一种平滑版 ReLU
        self.mlp = nn.Sequential(
                Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
            )

    def forward(
        self,
        x: Tensor,
        cu_seqlens = None
    ):
        x = x + self.attn(self.attn_ln(x), cu_seqlens=cu_seqlens)  # 注意力残差子层
        x = x + self.mlp(self.mlp_ln(x))  # 前馈残差子层
        return x


class WhisperEncoder(nn.Module):
    """
    Whisper 风格音频编码器整体结构：
        mel 频谱 [n_mels, T]
            → conv1 + GELU (stride=1)         时间维不变
            → conv2 + GELU (stride=2)         时间维减半
            → 加正弦位置编码
            → N 层 ResidualAttentionBlock     自注意力建模
            → AvgPool1d(stride=2)             时间维再减半 → 25Hz
            → LayerNorm + Linear 投影         输出 512 维特征
            → 首尾插入 BOS/EOS 特殊 token
    """
    def __init__(
            self,
            n_mels: int,       # mel 频谱的频段数（80 或 128）
            n_ctx: int,        # 位置编码最大长度（对应最长 mel 帧数）
            n_state: int,      # 模型隐藏维度（Transformer 的特征维）
            n_head: int,       # 注意力头数
            n_layer: int,      # Transformer 层数
            n_window: int = 1500,    # 单段处理的最大时间窗（帧数），超长音频会切块送入，避免一次吃太长序列爆显存
            output_dim: int = 512,   # 最终输出特征维度（送给 VQ 的 latent 维度）
            grad_checkpointing: bool = False,
            enable_mp: bool = False,
            audio_sequence_parallel: bool = False,
    ):
        super().__init__()
        # 第一层卷积：把 mel 频段投影到 n_state，时间维不变
        # 输入 [batch, n_mels, T] → 输出 [batch, n_state, T]
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        # 第二层卷积：stride=2，时间维减半（100 帧/秒 → 50 帧/秒）
        # 输入 [batch, n_state, T] → 输出 [batch, n_state, T/2]
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        # 注册正弦位置编码（buffer 不参与训练，只在前向时加到特征上）
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.n_layer = n_layer
        self.n_mels = n_mels

        # N 层 Transformer 编码器块
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, enable_mp=enable_mp, sequence_parallel=audio_sequence_parallel)
             for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)
        # 平均池化：stride=2，再把时间维减半（50 帧/秒 → 25 帧/秒，正好对应 25Hz tokenizer）
        self.avg_pooler = nn.AvgPool1d(2, stride=2)

        # 投影到输出维度
        self.proj = torch.nn.Linear(n_state, output_dim)  # [T, n_state] → [T, 512]

        # 每个音频段开头/结尾要插入两个特殊 token：
        #   BOS (Begin Of Sequence) = 音频段开始
        #   EOS (End Of Sequence) = 音频段结束
        # 用可学习 embedding 表示，让 LLM 知道一段音频的边界。
        self.audio_bos_eos_token = nn.Embedding(2, output_dim)

        self.output_dim = output_dim
        self.grad_checkpointing = grad_checkpointing
        self.enable_mp = enable_mp
        self.n_head = n_head
        self.n_state = n_state
        self.n_window = n_window

        self.audio_sequence_parallel = audio_sequence_parallel

        self.tp_world_size = 1

        self.set_audio_sync()

    def set_audio_sync(self):
        # 把除 Transformer blocks 之外的参数标记为"音频同步"。
        # 推测用于多卡/序列并行时的梯度同步策略：卷积和投影层在每张卡上保持一致，
        # 只有 blocks 内部按序列切分并行计算。
        for name, param in self.named_parameters():
            if not name.startswith("blocks"):
                setattr(param, "audio_sync", True)

    def forward(self, x_list: List[Tensor], audio_mellens:List[int], audio_aftercnnlens:List[int], audio_seqlens:List[int]):
        """
        x : torch.Tensor, shape = (n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        # 输入参数说明：
        #   x_list: 每条音频的 mel 频谱 list，每个元素形状 [n_mels, T_mel]
        #   audio_mellens: 每条音频的 mel 帧数（T_mel）
        #   audio_aftercnnlens: 每条音频卷积后的帧数（T_mel/2）
        #   audio_seqlens: 每条音频池化后的 token 帧数（T_mel/4，即 25Hz 帧数）
        aftercnn_x_list = []
        for each_x in x_list:
            # 超长音频按 n_window*2 切块，逐块过卷积。
            # 为什么不直接整条过？因为位置编码只有 n_ctx 长，且超长序列会爆显存。
            each_x_split_list = each_x.split(self.n_window * 2, dim=1)
            for each_x_split in each_x_split_list:
                each_x_split = F.gelu(self.conv1(each_x_split))  # [n_state, T_split]
                each_x_split = F.gelu(self.conv2(each_x_split))  # [n_state, T_split/2]
                # 转置成 (时间, 特征) 方便加位置编码：[T_split/2, n_state]
                each_x_split = each_x_split.permute(1, 0) # L,D
                # 取前 T_split/2 个位置编码，加到特征上
                each_positional_embedding_split = self.positional_embedding[:each_x_split.shape[0]]
                aftercnn_x_list.append(each_x_split+each_positional_embedding_split.to(each_x_split.dtype))

        # 把所有块沿时间维拼成一个长张量（packed 形式）：[总帧数, n_state]
        x = torch.cat(aftercnn_x_list, dim=0)
        src_len = x.size(0)

        # 构造 cu_seqlens：因为超长音频被切成了多块，每条音频可能对应多段。
        # 这里重新计算每一段（切块后）的长度，用于 flash-attn 知道段边界。
        output_list = []
        for item in audio_aftercnnlens:
            # item 是卷积后整条音频的帧数，按 n_window 切成多段
            while item > self.n_window:
                output_list.append(self.n_window)
                item -= self.n_window
            output_list.append(item)

        # 前缀和：[0, seg1_len, seg1_len+seg2_len, ...]，标出每段在长张量里的起止位置
        cu_seqlens = list(accumulate(output_list, func=operator.add,initial=0))
        cu_seqlens = torch.Tensor(cu_seqlens).to(device=x.device, dtype=torch.int32)

        layer_id = 0
        for block in self.blocks:
            layer_id+=1
            # 逐层过 Transformer 编码器，形状始终是 [总帧数, n_state]
            x = block(x, cu_seqlens=cu_seqlens)

        if self.avg_pooler:
            # 按每条音频的卷积后帧数切回各自序列，做时间维 2 倍平均池化（50Hz → 25Hz）
            x_list = x.split(audio_aftercnnlens, dim=0)
            token_x_list = []
            for x in x_list:
                x = x.permute(1, 0)  # [n_state, T_aftercnn]
                x = self.avg_pooler(x)  # [n_state, T_aftercnn/2]
                x = x.permute(1, 0)  # [T_aftercnn/2, n_state]
                token_x_list.append(x)
            # 拼回打包形式：[总 token 帧数, n_state]
            x = torch.cat(token_x_list, dim=0)

        x = self.ln_post(x)  # 最终 LayerNorm
        x = self.proj(x)     # 投影到 output_dim：[总 token 帧数, 512]

        # 在每条音频的 token 序列前后各插入一个 BOS/EOS 特殊 token。
        # 总输出长度 = 真实 token 数 + 每条音频多 2 个特殊 token。
        output = torch.zeros(
            (x.size(0) + len(audio_seqlens) * 2, x.size(1)),
            device=x.device, dtype=x.dtype
        )

        # 每条音频 token 区间的前缀和
        audio_seqlens_acc = list(accumulate(audio_seqlens, func=operator.add, initial=0))
        # 每条音频在 output 里的开头位置（要放 BOS）
        start_ids = torch.tensor(audio_seqlens_acc[:-1], device=x.device, dtype=torch.int32)
        # 每条音频在 output 里的结尾位置（要放 EOS）。注意 +2 是因为前面插了 BOS。
        end_ids = torch.tensor(audio_seqlens_acc[1:], device=x.device, dtype=torch.int32) - 1

        # 先把所有位置视为"真实 token"，再把首尾两个位置标成特殊 token 位置
        audio_tokens_mask = torch.ones(output.size(0), device=x.device, dtype=torch.bool)
        audio_tokens_mask[start_ids] = False
        audio_tokens_mask[end_ids] = False
        # 开头放 BOS embedding（index 0），结尾放 EOS embedding（index 1）
        output[start_ids] = self.audio_bos_eos_token.weight[0].to(x.dtype)
        output[end_ids] = self.audio_bos_eos_token.weight[1].to(x.dtype)
        # 其余位置填真实特征
        output[audio_tokens_mask] = x
        return output

    def lock(self, layers: int):
        # 冻结前若干层（卷积 + 前 layers 个 Transformer block）的参数，用于微调时只训高层。
        # 底层卷积学到的是通用音频特征（边缘、频谱纹理），高层学到的是任务相关特征。
        # 冻结底层可以省显存、防止小数据集过拟合。
        self.conv1.requires_grad_(False)
        self.conv2.requires_grad_(False)
        for i in range(min(layers, len(self.blocks))):
            self.blocks[i].requires_grad_(False)
