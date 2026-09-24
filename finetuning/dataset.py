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
#
# 本文件定义 TTSDataset —— Qwen3-TTS SFT 流程的"第二步：数据集类"。
# 它负责把 prepare_data.py 产出的 jsonl（每行已有 audio_codes）变成
# PyTorch DataLoader 能迭代、模型能直接吃的 batch 张量。
#
# 关键点：
#   - __getitem__：对每条样本，把目标文本包成"<|im_start|>assistant ... "格式，
#     tokenize 出 text_ids；把 ref_audio 读成波形并抽 mel 谱（用于说话人编码器）；
#     audio_codes 直接转 tensor。返回 dict。
#   - collate_fn：把一个 batch 的多条样本按最长序列 pad 成统一长度，
#     构造 talker 模型需要的 input_ids（双通道：文本/codec）、各种 mask、labels。
#     Qwen3-TTS 用 16 层码本并行预测，所以 codec_ids 维度是 [b, t, 16]。
#
# 输入数据每行字段：audio / text / audio_codes / language(可选) / ref_audio。
# 其中 audio_codes 由 prepare_data.py 预先用 tokenizer 离散化得到。
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

# 音频输入支持的形式：路径/URL/base64、numpy 波形（需配合 sr）、或 (waveform, sr) 元组。
AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]

class TTSDataset(Dataset):
    """Qwen3-TTS 训练用 Dataset。
    data_list：prepare_data.py 产出的 dict 列表（每条含 audio_codes）；
    processor：模型文本 tokenizer（用来把文本变成 token id）；
    config：Qwen3TTSConfig，提供各种特殊 token id（pad/bos/eos 等）。"""

    def __init__(self, data_list, processor, config:Qwen3TTSConfig, lag_num = -1):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config

    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        """用 librosa 读音频为单声道 float32 波形，返回 (audio, sr)。
        librosa 是常用的音频读写/分析库，sr=None 表示保留原始采样率。"""
        audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - np.ndarray: waveform (NOT allowed alone here because sr is unknown)
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    
    def _build_assistant_text(self, text: str) -> str:
        """把目标文本包成 Qwen3-TTS 训练时 assistant 通道的格式。
        末尾再起一个 <|im_start|>assistant 让模型从这里开始生成 codec。"""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _ensure_list(self, x: MaybeList) -> List[Any]:
        """标量也包成 list，统一后续处理。"""
        return x if isinstance(x, list) else [x]
    
    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        """用 processor（文本 tokenizer）把文本变成 input_ids 张量。"""
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id
    
    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        """抽 mel 谱图，喂给说话人编码器 (speaker_encoder) 得到 x-vector。
        Qwen3-TTS 要求 24kHz 音频；mel 参数与模型训练时一致。"""
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
        return mels



    def __getitem__(self, idx):
        """取一条样本，组装成单条 dict 供 collate_fn 进一步打包。
        返回字段：
          - text_ids：assistant 通道文本 token，去掉最后 5 个用于对齐；
          - audio_codes：[t, 16] 的离散音频码本（16 层码本）；
          - ref_mel：参考音频 mel 谱（用于说话人向量）。"""
        item = self.data_list[idx]

        audio_path  = item["audio"]
        text        = item["text"]
        audio_codes = item["audio_codes"]
        language        = item.get('language','Auto')
        ref_audio_path  = item['ref_audio']

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav,sr = normalized[0]

        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids":text_ids[:,:-5],    # 1 , t
            "audio_codes":audio_codes,      # t, 16
            "ref_mel":ref_mel
        }
        
    def collate_fn(self, batch):
        """把多条样本 pad 到同一长度并构造 talker 训练所需的双通道输入。
        Qwen3-TTS talker 输入是 [b, t, 2]：通道0放文本 token，通道1放 codec token；
        同时构造各种 mask（哪些位置参与 embedding/attention/loss）。
        codec_ids 是 [b, t, 16]：16 层码本用于多码本并行预测。
        codec_0_labels 用 -100 屏蔽非生成位置，是第 0 层码本的监督标签。"""
        assert self.lag_num == -1

        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b,t = len(batch),max_length

        input_ids   = torch.zeros((b,t,2),dtype=torch.long)
        codec_ids   = torch.zeros((b,t,16),dtype=torch.long)
        text_embedding_mask     = torch.zeros((b,t),dtype=torch.bool)
        codec_embedding_mask    = torch.zeros((b,t),dtype=torch.bool)
        codec_mask      = torch.zeros((b,t),dtype=torch.bool)
        attention_mask  = torch.zeros((b,t),dtype=torch.long)
        codec_0_labels  = torch.full((b, t), -100, dtype=torch.long)

        for i,data in enumerate(batch):
            text_ids        = data['text_ids']
            audio_codec_0   = data['audio_codes'][:,0]
            audio_codecs    = data['audio_codes']

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]
            
            # 通道 0（文本通道）布局：[BOS区3个] [pad 4个] [bos] [assistant文本去掉前3] [eos] [pad 直到 codec 结束]
            # 文本通道 mask 标记哪些位置用文本 embedding。
            input_ids[i,  :3, 0] = text_ids[0,:3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i,   7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0,3:]
            input_ids[i,   8+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len , 0] = self.config.tts_pad_token_id
            text_embedding_mask[i,  :8+text_ids_len+codec_ids_len] = True

            # 通道 1（codec 通道）布局：[空3] [nothink/think_bos/think_eos/0(pad spk)/pad] [pad] [pad] [codec_bos] [audio_codec_0...] [codec_eos]
            # 第 6 位留出来给 speaker embedding（见 sft_12hz.py 中 input_codec_embedding[:,6,:]=speaker_embedding）。
            # input_ids[i,   :3, 1] = 0
            input_ids[i,    3:8 ,1] = torch.tensor(
                                        [
                                            self.config.talker_config.codec_nothink_id,
                                            self.config.talker_config.codec_think_bos_id,
                                            self.config.talker_config.codec_think_eos_id,
                                            0,     # for speaker embedding
                                            self.config.talker_config.codec_pad_id       
                                        ]
                                    )
            input_ids[i,    8:8+text_ids_len-3  ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    8+text_ids_len-3    ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    8+text_ids_len-2    ,1] = self.config.talker_config.codec_bos_id
            input_ids[i,    8+text_ids_len-1:8+text_ids_len-1+codec_ids_len,    1] = audio_codec_0
            input_ids[i,    8+text_ids_len-1+codec_ids_len,    1] = self.config.talker_config.codec_eos_token_id

            # 监督标签：只在 codec 真实位置和 eos 处有效，其余 -100 被 loss 忽略。
            codec_0_labels[i,    8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i,    8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            # 16 层码本填到 codec_ids（第 0 层已在 input_ids 通道 1）。
            codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len,:] = audio_codecs

            codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 6] = False       # 第 6 位是 speaker embedding，不走 codec_embedding

            codec_mask[i,   8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :8+text_ids_len+codec_ids_len] = True
        
        # 把 batch 内各条的 ref_mel 在第 0 维拼起来，整体作为说话人编码器输入。
        ref_mels = [data['ref_mel'] for data in batch]
        ref_mels = torch.cat(ref_mels,dim=0)

        return {
            'input_ids':input_ids,
            'ref_mels':ref_mels,
            'attention_mask':attention_mask,
            'text_embedding_mask':text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask':codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels':codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask':codec_mask
        }