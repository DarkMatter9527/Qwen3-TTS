# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Core vector quantization implementation.
核心向量量化实现

本文件职责（在整体架构中的位置）
--------------------------------
25Hz 版（V1）语音 tokenizer 的 VQ 基础实现，被 `speech_vq.py` 的 `WhisperEncoderVQ` 调用。
负责把 Whisper encoder 输出的连续特征向量近似成离散编号（audio token）。

核心类层级
----------
- `EuclideanCodebook`     欧氏码本：用欧氏距离找最近邻的码本，含 kmeans 初始化 / EMA 更新 / 死码替换
- `VectorQuantization`     单层向量量化：包一层 codebook，加 commit loss 和直通估计
- `DistributedResidualVectorQuantization` (RVQ)：多层残差 VQ，第一层记大概，后面逐层逼近残差
- `DistributedGroupResidualVectorQuantization` (GRVQ)：RVQ 的分组变体，把特征分若干组各做 RVQ

关键术语通俗解释
----------------
- 码本 codebook：一本"标准声音碎片字典"，每项是一个固定维度向量；输入特征去字典里找最像的项
- VQ 向量量化：把连续向量近似成码本里"最近邻"项编号的过程（找最近邻）
- RVQ 残差向量量化：多本字典层层逼近——第一本记大概，后面每本记越来越细的"误差"
- GRVQ 分组残差 VQ：RVQ 的分组变体，先把特征切若干组，每组分别做 RVQ
- commit loss 承诺损失：训练时把输入特征"拉"向码本项的 MSE 损失
- straight-through estimator 直通估计：VQ 反向传播技巧——前向用硬编号（码本项）替换输入，
  反向时让梯度"直通"传回连续特征（因为 argmax 不可导）
- EMA 指数移动平均：码本更新方式之一，不靠梯度反传，而用加权移动平均更新码本向量
- 死码 dead code：长期不被选中的码本项；用当前 batch 的随机样本替换以避免码本坍缩
"""
import random
import typing as tp
from random import randrange

import numpy as np
from einops import rearrange, repeat
from math import ceil
import torch
from torch import nn
import torch.nn.functional as F


def round_up_multiple(num, mult):
    # 把 num 向上取整到 mult 的整数倍（用于对齐序列长度）
    return ceil(num / mult) * mult

def default(val: tp.Any, d: tp.Any) -> tp.Any:
    # val 为 None 时返回默认值 d，否则返回 val
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    # 原地 EMA 更新：moving_avg = decay*moving_avg + (1-decay)*new
    # decay 越大，历史惯性越强（码本更新越慢越稳）
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    # 拉普拉斯平滑：(x+eps)/(x.sum()+n*eps)，避免某类为 0 导致除零
    # 用于码本 EMA 时归一化 cluster_size，使每个码本项贡献稳定
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    # 用 Kaiming 均匀分布初始化一个空张量（用于不跑 kmeans 时的码本初值）
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    # 从 samples 中随机抽取 num 个向量，用于码本初始化或死码替换
    # 样本足够多时无放回采样；不足时有放回采样
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


@torch.no_grad()
def kmeans(samples, num_clusters: int, num_iters: int = 10):
    # 简易 kmeans，用于码本初始化：从 samples 中选 num_clusters 个初始中心，
    # 跑 num_iters 轮"找最近中心→更新中心"。返回最终中心 (num_clusters, dim) 和每个簇的大小。
    dim, dtype = samples.shape[-1], samples.dtype

    # 初始中心从 samples 中随机抽
    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        # 计算每个 sample 到每个 mean 的（负）欧氏距离，等价于 ||x-m||^2 = x^2 - 2xm + m^2
        # 取 max（因为是负距离）即等价于取最小距离，得到每个 sample 所属簇
        dists = -(
                samples.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(samples, means.t())
                + means.t().pow(2).sum(0, keepdim=True)
        )

        buckets = dists.max(dim=-1).indices
        del dists
        # 统计每个簇的样本数
        bins = torch.bincount(buckets, minlength=num_clusters)
        # 空簇分母设为 1 防除零
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        # 累加每个簇内的样本得到新中心
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        # 空簇保持旧中心，非空簇用新中心
        means = torch.where(zero_mask[..., None], means, new_means)
    return means, bins


def preprocess(x):
    # 把任意带 ... 的前导维度合并成一维，只保留最后一维特征维
    # 即 (... , d) -> (N, d)，便于在码本上批量找最近邻
    x = rearrange(x, "... d -> (...) d")
    return x


def postprocess_emb(embed_ind, shape):
    # 把编号张量还原成输入的形状（去掉特征维 d），便于后续使用
    return embed_ind.view(*shape[:-1])


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    欧氏码本：用欧氏距离找最近邻的码本实现。

    通俗解释
    --------
    存一本"标准声音碎片字典"（embed: shape=(codebook_size, dim)），给定一个连续向量，
    去字典里找欧氏距离最小的那一项，返回它的编号。训练时还会用 EMA 更新字典内容，
    并把长期没人用的"死码"用当前 batch 的样本替换掉，避免字典坍缩。

    Args:
        dim (int): 维度（每个码本向量的维度）。
        codebook_size (int): 码本大小（字典里有多少项）。
        kmeans_init (bool): 是否用 k-means 初始化码本。若为 True，在第一个训练 batch 上
            跑 kmeans，用得到的聚类中心作为码本初值。
        kmeans_iters (int): k-means 初始化时的迭代次数。
        decay (float): 码本 EMA 更新的衰减系数，越大更新越慢越稳。
        epsilon (float): 数值稳定用的小常数。
        threshold_ema_dead_code (int): 死码阈值。任何 EMA 簇大小小于该阈值的码本项
            会被当前 batch 中随机抽取向量替换。
    """

    def __init__(
            self,
            dim: int,
            codebook_size: int,
            kmeans_init: int = False,
            kmeans_iters: int = 10,
            decay: float = 0.99,
            epsilon: float = 1e-5,
            threshold_ema_dead_code: float = 2.0,
    ):
        super().__init__()
        self.decay = decay
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        # 以下 buffer 在外层（RVQ）集中注册并传入，便于多卡同步；本类只是占位
        self.inited = None
        self.cluster_size = None
        self.embed = None
        self.embed_avg = None
        self.training = True

    def init_embed_(self, data):
        # 在第一次前向时用 kmeans 初始化码本（仅一次）。之后各 buffer 由外层同步。
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        # distrib.broadcast_tensors([self.embed, self.embed_avg, self.cluster_size, self.inited])

    def replace_(self, samples, mask):
        # 用 samples 中随机取向量替换 mask 为 True 的码本项（死码替换）
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        # 死码过期检测：把长期使用率太低的码本项替换成 batch 中的随机样本。
        if self.threshold_ema_dead_code == 0:
            return

        # 把 cluster_size 归一化到码本规模，便于和 threshold 比较
        cluster_size = self.cluster_size / sum(self.cluster_size) * self.codebook_size
        expired_codes = cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        else:
            print(f"VQ expire infos: num_expire={sum(expired_codes)}, cluster_size[:5]={cluster_size[:5]}")

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        # sync buffers outside for efficiency
        # distrib.broadcast_tensors(self.buffers())

    def quantize(self, x):
        # 核心：在码本中找最近邻，返回每条输入向量对应的编号
        # 用负欧氏距离等价表达：-||x-e||^2 = -(x^2 - 2xe + e^2)
        # max（负距离）= 最小距离，indices 即最近邻的码本项下标
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def dequantize(self, embed_ind):
        # 编号 -> 码本向量：用 embedding 查表把编号还原成码本里的向量
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x, buffers):
        # 编码：连续向量 -> 编号（推理用，不更新码本）
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers

        shape = x.shape
        # 预处理：合并前导维度
        x = preprocess(x)
        # 找最近邻编号
        embed_ind = self.quantize(x)
        # 还原形状（去掉特征维）
        embed_ind = postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind, buffers):
        # 解码：编号 -> 码本向量（推理用）
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers

        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x, buffers):
        # 训练前向：找最近邻 + 更新码本 + 返回量化结果和编号
        self.inited, self.cluster_size, self.embed, self.embed_avg = buffers

        shape, dtype = x.shape, x.dtype
        x = preprocess(x)

        # 首次前向时用 kmeans 初始化码本
        self.init_embed_(x)
        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            # 在码本已同步时检测死码，保证各 worker 决策一致
            self.expire_codes_(x)

        # 找最近邻编号
        embed_ind = self.quantize(x)
        # one-hot 用于后续 EMA 统计
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = postprocess_emb(embed_ind, shape)
        # 编号 -> 量化向量
        quantize = self.dequantize(embed_ind)

        if self.training:
            # === EMA 更新码本（不靠梯度，靠统计）===
            # 更新 cluster_size（每项被选中的频次）
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            # 更新 embed_avg（每项对应的输入向量平均）
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            # 用拉普拉斯平滑后的 cluster_size 归一化 embed_avg，得到新码本
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            # Note: after ema update, there is a very small difference between codebooks on GPUs.
            # The impact can be very small, ignore it.

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    单层向量量化实现（包在 EuclideanCodebook 之上）。

    通俗解释
    --------
    在 `EuclideanCodebook` 之外加两层功能：
    1. 可选维度投影（codebook_dim 与 dim 不一致时先 project_in 再量化，结束再 project_out）
    2. 训练时构造 commit loss（让输入特征"承诺"靠近码本项）+ straight-through 估计
       （前向用码本向量替换输入，反向让梯度从量化结果直通到输入特征）

    Currently, supports only euclidean distance.
    Args:
        dim (int): 维度
        codebook_size (int): 码本大小
        codebook_dim (int): 码本维度。未指定时用 dim。
        decay (float): 码本 EMA 更新的衰减系数。
        epsilon (float): 数值稳定用小常数。
        kmeans_init (bool): 是否用 kmeans 初始化码本。
        kmeans_iters (int): kmeans 初始化迭代次数。
        threshold_ema_dead_code (int): 死码阈值。
        commitment_weight (float): commit loss 的权重。
    """
    def __init__(
            self,
            dim: int,
            codebook_size: int,
            codebook_dim: tp.Optional[int] = None,
            decay: float = 0.99,
            epsilon: float = 1e-5,
            kmeans_init: bool = True,
            kmeans_iters: int = 50,
            threshold_ema_dead_code: float = 2.0,
            commitment_weight: float = 1.,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        # 码本维度和输入维度不一致时加线性投影对齐
        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim)) if requires_projection else (nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim)) if requires_projection else (nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        # 内部欧氏码本
        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size
        self.training = True

    @property
    def codebook(self):
        # 暴露内部码本矩阵 (codebook_size, dim)
        return self._codebook.embed

    def encode(self, x, buffers):
        # 编码：连续向量 -> 编号（推理路径）
        # x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x, buffers)
        return embed_in

    def decode(self, embed_ind, buffers):
        # 解码：编号 -> 连续向量（推理路径）
        quantize = self._codebook.decode(embed_ind, buffers)
        quantize = self.project_out(quantize)
        # quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x, buffers):
        # 训练前向：量化 + commit loss + straight-through 估计
        device = x.device
        # x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        # 调用 EuclideanCodebook 得到量化结果和编号
        quantize, embed_ind = self._codebook(x, buffers)

        if self.training:
            # === straight-through estimator 直通估计 ===
            # 前向：quantize 用码本向量（硬编号结果）
            # 反向：(quantize - x).detach() 让差值不传梯度，最终 quantize 对 x 的梯度 = 1
            # 等效于梯度直接从 quantize 传回 x，绕过不可导的 argmax
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                # === commit loss 承诺损失 ===
                # MSE(quantize.detach(), x)：把输入 x 拉向被选中的码本项
                # 注意 quantize 已 detach，梯度只回传给 x（码本由 EMA 更新，不走梯度）
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        # quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class DistributedResidualVectorQuantization(nn.Module):
    """Efficient distributed residual vector quantization implementation.
    分布式残差向量量化（RVQ）实现。

    通俗解释
    --------
    多层 VQ 叠加：
    - 第 1 层量化原始输入，得到编号 1 和量化结果 q1，残差 r1 = x - q1
    - 第 2 层量化残差 r1，得到编号 2 和 q2，残差 r2 = r1 - q2
    - ……
    第一本字典记"大概"，后面每本字典记越来越细的"误差"，多层加起来逼近原输入。
    最终输出多个编号（每层一个）和累加的量化结果。

    支持 quantize_dropout：训练时随机丢弃若干高层量化器，让低层学会独立编码
    （类似 SoundStream/EnCodec 的 dropout，提升码本利用率）。

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *,
                 num_quantizers,
                 quantize_dropout: bool = False,
                 rand_num_quant: tp.Optional[tp.List] = None,
                 **kwargs):
        super().__init__()
        """
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        """
        # 码本维度若未指定则等于输入 dim
        codebook_size, codebook_dim = kwargs["codebook_size"], kwargs["codebook_dim"] if kwargs["codebook_dim"] else kwargs["dim"]
        kmeans_init = kwargs["kmeans_init"]
        if isinstance(kmeans_init, bool):
            if not kwargs["kmeans_init"]:
                # 不跑 kmeans：用 Kaiming 均匀初始化，且标记已初始化
                embed = uniform_init(num_quantizers, codebook_size, codebook_dim)
                inited = True
            else:
                # 要跑 kmeans：先占位为 0，等第一个 batch 再初始化
                embed = torch.zeros(num_quantizers, codebook_size, codebook_dim)
                inited = False
        elif isinstance(kmeans_init, str):
            # 从 .npy 文件加载预先算好的 kmeans 中心
            embed = np.load(kmeans_init)
            embed = torch.from_numpy(embed)
            if embed.dim() == 2:
                # 单层 → 扩展为多层
                embed = embed.unsqueeze(0)
            inited = True
        else:
            raise TypeError("kmeans_init should be either a bool or string path to init weights.")

        # 集中注册 buffer（inited / cluster_size / embed / embed_avg），
        # 便于多卡 all-reduce / broadcast 同步
        self.register_buffer("inited", torch.Tensor([[inited] for _ in range(num_quantizers)]))
        self.register_buffer("cluster_size", torch.zeros(num_quantizers, codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

        # 第 0 层可设独立下采样倍率（默认 1）
        self.q0_ds_ratio = 1
        if "q0_ds_ratio" in kwargs:
            self.q0_ds_ratio = kwargs.pop("q0_ds_ratio")

        # 每层一个 VectorQuantization
        self.layers = nn.ModuleList()
        for i in range(num_quantizers):
            vq_args = dict(**kwargs)
            vq = VectorQuantization(**vq_args)
            self.layers.append(vq)

        self.quantize_dropout = quantize_dropout
        self.rand_num_quant = rand_num_quant

    def forward(self, x, n_q: tp.Optional[int] = None):
        # RVQ 训练前向：多层量化 + 残差传递 + 可选 dropout
        quantized_out = torch.zeros_like(x)
        residual = x
        bb, cc, tt = x.shape
        device = x.device

        all_losses = []
        all_indices = []
        all_sub_quants = []
        # 实际使用的量化层数（默认全部）
        n_q = n_q or len(self.layers)

        # 是否对高层量化器做随机丢弃（仅训练时）
        should_quantize_dropout = self.training and self.quantize_dropout and self.rand_num_quant is not None
        if should_quantize_dropout:
            # 随机选一个起始层，该层之后的量化器全部"跳过"
            rand_quantize_dropout_index = random.choice(self.rand_num_quant)

            null_indices_shape = (x.shape[0], x.shape[2])
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)
            null_loss = torch.full((1,), 0., device=device, dtype=x.dtype)
            null_sub_quant = torch.full(x.shape, -1, device=device, dtype=x.dtype)

        for quantizer_index, layer in enumerate(self.layers[:n_q]):
            # dropout except the first quantizer
            # 第 0 层必须量化，后续层可被 dropout 跳过
            if should_quantize_dropout and quantizer_index >= rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                all_sub_quants.append(null_sub_quant)
                continue

            # 第 0 层如果设了下采样倍率，先把时间维下采样再量化（省算力 / 提升感受野）
            quant_in = residual
            if self.q0_ds_ratio > 1 and quantizer_index == 0:
                quant_in = F.interpolate(quant_in, size=[tt//2])
            # 单层量化
            quantized, indices, loss = layer(quant_in, [
                self.inited[quantizer_index],
                self.cluster_size[quantizer_index],
                self.embed[quantizer_index],
                self.embed_avg[quantizer_index]
            ])
            if self.q0_ds_ratio > 1 and quantizer_index == 0:
                # 把第 0 层结果上采样回原长度
                quantized = F.interpolate(quantized, size=[tt])
                indices = F.interpolate(indices.unsqueeze(1).float(), size=[tt]).squeeze(1).long()
            # 残差传递：下一层量化"原输入 - 当前量化结果"
            residual = residual - quantized
            # 累加各层量化结果
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
            all_sub_quants.append(quantized)

        # sync buffers after one forward step
        # distrib.broadcast_tensors(self.buffers())
        # 沿"层"维堆叠：indices 变为 (num_quantizers, B, T)
        out_losses, out_indices, out_sub_quants = map(torch.stack, (all_losses, all_indices, all_sub_quants))

        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        # 推理编码：多层残差量化，只输出编号（每层一个）
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for i, layer in enumerate(self.layers[:n_q]):
            # 当前层找编号
            indices = layer.encode(residual, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            # 用编号解出量化结果，更新残差
            quantized = layer.decode(indices, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            residual = residual - quantized
            all_indices.append(indices)
        # (num_quantizers, B, T)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        # 推理解码：多层编号 -> 累加的连续向量
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            # 每层编号解出量化结果并累加
            quantized = layer.decode(indices, [
                self.inited[i],
                self.cluster_size[i],
                self.embed[i],
                self.embed_avg[i]
            ])
            quantized_out = quantized_out + quantized
        return quantized_out


class DistributedGroupResidualVectorQuantization(nn.Module):
    """Efficient distributed group residual vector quantization implementation.
    分布式分组残差向量量化（GRVQ）实现。

    通俗解释
    --------
    RVQ 的分组变体：先把特征在通道维切若干组（num_groups），每组独立做 RVQ。
    相比单条 RVQ，分组可以在不增加每层码本大小的情况下提升表达能力
    （不同子空间各自学习码本）。

    Follows Algorithm 1. in https://arxiv.org/abs/2305.02765
    Group Then rvq
    """
    def __init__(self, *,
                 num_groups,
                 num_quantizers,
                 quantize_dropout: bool = False,
                 rand_num_quant: tp.Optional[tp.List] = None,
                 **kwargs):
        super().__init__()
        # 每组一个独立的 RVQ
        self.rvqs = nn.ModuleList(
            [
                DistributedResidualVectorQuantization(
                    num_quantizers=num_quantizers,
                    quantize_dropout=quantize_dropout,
                    rand_num_quant=rand_num_quant,
                    **kwargs
                )
                for _ in range(num_groups)
            ]
        )
        self.num_groups = num_groups

    def forward(self, x, n_q: tp.Optional[int] = None):
        # GRVQ 训练前向：通道维分组 → 每组 RVQ → 拼回
        # 沿 dim=1 切成 num_groups 份
        x_lst = torch.chunk(x, chunks=self.num_groups, dim=1)
        all_quantized_out = []
        all_indices = []
        all_losses = []
        for mod, item in zip(self.rvqs, x_lst):
            # 每组独立 RVQ
            quantized_out, out_indices, out_losses = mod(item, n_q)
            all_quantized_out.append(quantized_out)
            all_indices.append(out_indices)
            all_losses.append(out_losses)

        # 各组 loss 求平均
        out_losses = torch.stack(all_losses, dim=1).mean(dim=1)

        # 量化结果沿通道维拼回，indices 增加一组维
        return torch.cat(all_quantized_out, dim=1), torch.stack(all_indices, dim=1), out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        # 推理编码：分组 → 每组 RVQ 编码 → 沿组维堆叠
        x_lst = torch.chunk(x, chunks=self.num_groups, dim=1)
        return torch.stack([mod.encode(item, n_q) for mod, item in zip(self.rvqs, x_lst)], dim=1)

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        # 推理解码：分组编号 → 每组 RVQ 解码 → 沿通道维拼回
        q_indices_lst = torch.chunk(q_indices, chunks=self.num_groups, dim=1)
        return torch.cat([mod.decode(item.squeeze(1)) for mod, item in zip(self.rvqs, q_indices_lst)], dim=1)