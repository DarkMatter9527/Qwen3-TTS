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
本文件职责（在整体架构中的位置）
--------------------------------
25Hz 版（V1）语音 tokenizer 的"语音专用 VQ 模块"层：在 `core_vq.py` 的通用 VQ 之上，
针对语音场景做工程封装：
- `MelSpectrogramFeatures`：BigVGAN 风格 mel 频谱图提取（音频波形 → mel）
- `XVectorExtractor`：基于 ONNX 的说话人 x-vector 提取器（用于条件控制音色）
- `WhisperEncoderVQ`：继承 `WhisperEncoder`，在 Transformer 某层后插入 GRVQ 量化，
  把连续特征离散成 audio token（每秒 25 步，每步 1 个编号，码本 32768 项）

术语速查
--------
- mel 频谱图：声音时间×频率的二维表示，由 STFT + mel 滤波器组得到
- x-vector：经典说话人嵌入（DNN 提取），用于条件生成
- GRVQ 分组残差 VQ：见 core_vq.py 注释
- Whisper encoder：OpenAI Whisper 模型的编码器部分
- 因果卷积 / 下采样 / 上采样：用 Conv1d/ConvTranspose1d 控制时间分辨率
"""
import sox
import copy
import torch
import operator
import onnxruntime

import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi

from librosa.filters import mel as librosa_mel_fn
from itertools import accumulate
from typing import List
from torch import Tensor

from .core_vq import DistributedGroupResidualVectorQuantization
from .whisper_encoder import WhisperEncoder, Conv1d, ConvTranspose1d


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    # 动态范围压缩：log(clamp(x))，把线性功率谱压到对数尺度，更接近人耳感知
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    # 谱归一化：对幅度谱做动态范围压缩（取对数）
    output = dynamic_range_compression_torch(magnitudes)
    return output

class MelSpectrogramFeatures(nn.Module):
    """
    Calculate the BigVGAN style mel spectrogram of an input signal.
    BigVGAN 风格的 mel 频谱图提取器。

    通俗解释
    --------
    把音频波形 (1, T) 通过"短时傅里叶变换 (STFT) → mel 滤波器组 → 取对数"
    转成 mel 频谱图 (n_mel_channels, n_frames)，作为后续 encoder 的输入特征。
    BigVGAN 风格的参数（hop/win/n_fft 等）需与训练时保持一致。

    Args:
        filter_length (int): STFT 的窗长（FFT 点数），默认 1024
        hop_length (int): 相邻帧之间的步长（采样点数），决定时间分辨率，默认 160
        win_length (int): 实际窗函数长度，默认 640
        n_mel_channels (int): mel 通道数（频率分辨率），默认 80
        mel_fmin (int): mel 频率下限，默认 0
        mel_fmax (int): mel 频率上限，默认 8000
        sampling_rate (int): 输入音频采样率，默认 16000
        sampling_rate_org (int, optional): 原始采样率（用于重采样时记录），默认 None
        padding (str): 填充模式，'center' 居中对称填，默认 'center'
 
    Returns:
        torch.Tensor: Mel spectrogram.
    """
    def __init__(self, 
                 filter_length=1024,
                 hop_length=160,
                 win_length=640,
                 n_mel_channels=80,
                 mel_fmin=0,
                 mel_fmax=8000,
                 sampling_rate=16000,
                 sampling_rate_org=None,
                 padding='center',
                 use_db = False,
                 ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
    
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        # 原始采样率未指定时与 sampling_rate 一致
        self.sampling_rate_org = sampling_rate_org if sampling_rate_org is not None else sampling_rate
        # mel 滤波器矩阵和 hann 窗按设备缓存（避免重复创建）
        self.mel_basis = {}
        self.hann_window = {}

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        # 不计算梯度（mel 提取是确定的可逆预处理）
        with torch.no_grad():
            feats = self.extract(audio, **kwargs) 
        return feats
    
    def extract(self, audio, **kwargs):
        # 把音频波形提取为 mel 频谱图（频率维 × 时间维）

        # 输入可能是 (B,1,T) 或 (B,T) 等，统一压到 (B,T)
        if len(audio.shape) == 3:
            audio = audio.squeeze(1) if audio.shape[1] == 1 else audio.squeeze(2)
        assert len(audio.shape) == 2

        y = audio
        # 懒加载：第一次按采样率和设备缓存 mel_basis 和 hann_window
        if len(list(self.mel_basis.keys())) == 0:
            mel = librosa_mel_fn(sr=self.sampling_rate, n_fft=self.filter_length, n_mels=self.n_mel_channels, fmin=self.mel_fmin, fmax=self.mel_fmax)
            self.mel_basis[str(self.mel_fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
            self.hann_window[str(y.device)] = torch.hann_window(self.win_length).to(y.device)

        # 'center' 填充：左右对称反射补零，保证 STFT 帧对齐
        y = torch.nn.functional.pad(y.unsqueeze(1), (int((self.filter_length-self.hop_length)/2), int((self.filter_length-self.hop_length)/2)), mode='reflect')
        y = y.squeeze(1)

        # STFT → 复数谱 → 取幅度
        spec = torch.stft(y, self.filter_length, hop_length=self.hop_length, win_length=self.win_length, window=self.hann_window[str(y.device)],
                          center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

        # mel 滤波器组：把线性频率维压到 mel 频率维
        spec = torch.matmul(self.mel_basis[str(self.mel_fmax)+'_'+str(y.device)], spec)
        # 谱归一化（取对数，压缩动态范围）
        spec = spectral_normalize_torch(spec)
    
        return spec
        

class XVectorExtractor(nn.Module):
    def __init__(self, audio_codec_with_xvector):
        super().__init__()
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.ort_session = onnxruntime.InferenceSession(audio_codec_with_xvector, sess_options=option, providers=providers)

        self.tfm = sox.Transformer()
        self.tfm.norm(db_level=-6)

        self.mel_ext = MelSpectrogramFeatures(
            filter_length=1024,
            hop_length=160,
            win_length=640,
            n_mel_channels=80,
            mel_fmin=0,
            mel_fmax=8000,
            sampling_rate=16000
        )

    def extract_code(self, audio):
        with torch.no_grad():
            norm_audio = self.sox_norm(audio)

            norm_audio = torch.from_numpy(copy.deepcopy(norm_audio)).unsqueeze(0)
            feat = kaldi.fbank(norm_audio,
                            num_mel_bins=80,
                            dither=0,
                            sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            norm_embedding = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten()
            norm_embedding = F.normalize(torch.from_numpy(norm_embedding), dim=0)
            
            ref_mel = self.mel_ext.extract(audio=norm_audio)
        
        return norm_embedding.numpy(), ref_mel.permute(0,2,1).squeeze(0).numpy()
    
    def sox_norm(self, audio):
        wav_norm = self.tfm.build_array(input_array=audio, sample_rate_in=16000)
        return wav_norm


class WhisperEncoderVQ(WhisperEncoder):
    def __init__(
            self,
            n_mels: int,
            n_ctx: int,
            n_state: int,
            n_head: int,
            n_layer: int,
            n_window: int = 1500,
            output_dim: int = 512,
            grad_checkpointing: bool = False,
            enable_mp: bool = False,
            audio_sequence_parallel: bool = False,
            audio_vq_layers: int = -1,
            audio_vq_type: str = "NULL",
            audio_vq_codebook_size: int = 4096,
            audio_vq_pe: bool = False,
            audio_vq_commit_loss: float = 0.0,
            audio_vq_out_commit_loss: float = 0.0,
            audio_vq_no_quantize: bool = False,
            audio_vq_ff_layer: int = 0,
            audio_vq_threshold_ema_dead_code: float = 0.1,
            audio_vq_codebook_dim: int = None,
            audio_vq_ds_rate: int = None,
    ):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer, n_window, output_dim, grad_checkpointing, enable_mp, audio_sequence_parallel)

        self.audio_vq_layers = audio_vq_layers
        self.audio_vq_type = audio_vq_type
        self.audio_vq_codebook_size = audio_vq_codebook_size
        self.audio_vq_pe = audio_vq_pe
        self.audio_vq_commit_loss = audio_vq_commit_loss
        self.audio_vq_out_commit_loss = audio_vq_out_commit_loss
        self.audio_vq_no_quantize = audio_vq_no_quantize
        self.audio_vq_ff_layer = audio_vq_ff_layer

        if audio_vq_layers > 0:
            self.vq_feature_dim = self.n_state
            self.audio_vq_ds_rate = 1
        else:
            raise NotImplementedError(f"Unsupported audio_vq_layers: {audio_vq_layers}")
        
        if self.audio_vq_ds_rate == audio_vq_ds_rate:
            self.audio_vq_downsample = nn.Identity()
            self.audio_vq_upsample   = nn.Identity()
        else:
            assert audio_vq_ds_rate % self.audio_vq_ds_rate == 0
            stride = audio_vq_ds_rate // self.audio_vq_ds_rate
            self.audio_vq_downsample = Conv1d(self.vq_feature_dim, self.vq_feature_dim, kernel_size=stride, stride=stride)
            self.audio_vq_upsample = ConvTranspose1d(self.vq_feature_dim, self.vq_feature_dim, kernel_size=stride, stride=stride)
            self.audio_vq_ds_rate = audio_vq_ds_rate
        
        if audio_vq_type == "GRVQ":
            self.audio_quantizer = DistributedGroupResidualVectorQuantization(
                codebook_size = audio_vq_codebook_size,
                dim = self.vq_feature_dim, 
                codebook_dim = self.vq_codebook_dim if audio_vq_codebook_dim is None else audio_vq_codebook_dim, 
                num_groups=1,
                num_quantizers=1,
                kmeans_init=False,
                threshold_ema_dead_code = audio_vq_threshold_ema_dead_code
            )
        else:
            raise NotImplementedError(f"Unsupported audio_vq_type: {audio_vq_type}")
        
        if self.audio_vq_pe:
            self.project_after_vq_pe = nn.Linear(self.n_state, self.n_state)

    def _calc_quantize_activities(self, indices):
        indices_onehot = F.one_hot(indices.long().flatten(), self.audio_vq_codebook_size).sum(dim=0)
        vq_num_activities = sum(indices_onehot>0)
        vq_num_tokens = sum(indices_onehot)
        return {
            "vq_num_activities": vq_num_activities,
            "vq_num_tokens": vq_num_tokens,
        }
    
    def _do_quantize(self, x, pe=None, y=None):
        """
            x: torch.Tensor, shape = (T, D)
            q: torch.Tensor, shape = (T, D)
            i: torch.Tensor, shape = (T)
        """
        if self.audio_vq_out_commit_loss > 0:
            x_teacher = x.clone()
        x = x.unsqueeze(0)

        x = self.audio_vq_downsample(x.transpose(1, 2))
        x = x.transpose(1, 2)

        vq_stats = {}

        if self.audio_vq_type == "GRVQ":
            if self.training:
                raise NotImplementedError
            else:
                indices = self.audio_quantizer.encode(x)
                x = self.audio_quantizer.decode(indices)
                indices = indices.squeeze(2).squeeze(1)

        vq_stats.update(self._calc_quantize_activities(indices))

        x, indices = x.squeeze(0), indices.squeeze(0)
        if self.audio_vq_pe:
            x = x + pe
            x = self.project_after_vq_pe(x)
        
        x = self.audio_vq_upsample(x.unsqueeze(0).transpose(1, 2))
        x = x.transpose(1, 2).squeeze(0)

        if self.audio_vq_out_commit_loss > 0:
            vq_out_commit_loss = F.mse_loss(x_teacher.detach(), x)
            vq_stats["vq_out_commit_loss"] = vq_out_commit_loss * self.audio_vq_out_commit_loss

        return x, indices, vq_stats

    def forward(self, x_list: List[Tensor], audio_mellens:List[int], audio_aftercnnlens:List[int], audio_seqlens:List[int], return_indices=False, audio_pitchs=None):
        """
        x : torch.Tensor, shape = (n_mels, n_ctx)
            the mel spectrogram of the audio
        """

        aftercnn_x_list = []
        pe_for_vq_list = []
        for each_x in x_list:
            each_x_split_list = each_x.split(self.n_window * 2, dim=1)
            for each_x_split in each_x_split_list:
                each_x_split = F.gelu(self.conv1(each_x_split))
                each_x_split = F.gelu(self.conv2(each_x_split))
                each_x_split = each_x_split.permute(1, 0) # L,D

                each_positional_embedding_split = self.positional_embedding[:each_x_split.shape[0]]
                aftercnn_x_list.append(each_x_split+each_positional_embedding_split.to(each_x_split.dtype))

                pe_for_vq_split = self.positional_embedding[:each_x_split.shape[0] // self.audio_vq_ds_rate]
                pe_for_vq_list.append(pe_for_vq_split.to(each_x_split.dtype))

        pe_for_vq = torch.cat(pe_for_vq_list, dim=0)
        x = torch.cat(aftercnn_x_list, dim=0)
        src_len = x.size(0)

        output_list = []
        for item in audio_aftercnnlens:
            while item > self.n_window:
                output_list.append(self.n_window)
                item -= self.n_window
            output_list.append(item)

        cu_seqlens = list(accumulate(output_list, func=operator.add,initial=0))
        cu_seqlens = torch.Tensor(cu_seqlens).to(device=x.device, dtype=torch.int32)

        layer_id = 0

        for block in self.blocks:
            layer_id+=1

            x = block(x, cu_seqlens=cu_seqlens)
            
            if self.audio_vq_layers == layer_id: # vq inside encoder
                x, indices, vq_stats = self._do_quantize(x, pe_for_vq)
                if return_indices:
                    return x, indices

        if self.avg_pooler:
            x_list = x.split(audio_aftercnnlens, dim=0)
            token_x_list = []
            for x in x_list:
                x = x.permute(1, 0)
                x = self.avg_pooler(x)
                x = x.permute(1, 0)
                token_x_list.append(x)
            x = torch.cat(token_x_list, dim=0)

        x = self.ln_post(x)

        x = self.proj(x)

        output = torch.zeros(
            (x.size(0) + len(audio_seqlens) * 2, x.size(1)),
            device=x.device, dtype=x.dtype
        )

        audio_seqlens_acc = list(accumulate(audio_seqlens, func=operator.add, initial=0))
        start_ids = torch.tensor(audio_seqlens_acc[:-1], device=x.device, dtype=torch.int32)
        end_ids = torch.tensor(audio_seqlens_acc[1:], device=x.device, dtype=torch.int32) - 1

        audio_tokens_mask = torch.ones(output.size(0), device=x.device, dtype=torch.bool)
        audio_tokens_mask[start_ids] = False
        audio_tokens_mask[end_ids] = False
        output[start_ids] = self.audio_bos_eos_token.weight[0].to(x.dtype)
        output[end_ids] = self.audio_bos_eos_token.weight[1].to(x.dtype)
        output[audio_tokens_mask] = x

        if self.audio_vq_type != "NULL":
            return output, vq_stats
        return output