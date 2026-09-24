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
# ====================================================================
# 文件总览：qwen3_tts_tokenizer.py
# --------------------------------------------------------------------
# 本文件是 Qwen3-TTS 语音 tokenizer 的"高层 API"封装层。
# 语音 tokenizer 是 TTS 系统的"声码器"部分，负责两件事：
#   - encode：把音频波形 → 离散码（codec tokens），供 LLM 当作"文本"处理；
#   - decode：把 LLM 生成的离散码 → 音频波形。
#
# 支持两种 tokenizer（由底层 model.config.model_type 决定）：
#   - 25Hz（V1）：每秒 25 个码，码为一维 (codes_len,)；decode 还需
#                 说话人向量 xvectors 和参考梅尔频谱 ref_mels 作为条件。
#   - 12Hz（V2）：每秒 12 个码，码为二维 (codes_len, num_quantizers)，
#                 多量化器残差编码；decode 只需 audio_codes。
#
# 主要导出：Qwen3TTSTokenizer
#   - from_pretrained(path, **kwargs)：HuggingFace 风格加载，注册两种 tokenizer 类型。
#   - encode(audios, sr, return_dict) ：批量编码音频为离散码。
#   - decode(encoded)                  ：把离散码解码回波形 (List[np.ndarray], sr)。
#   - load_audio / get_*_sample_rate 等辅助方法。
#
# 整体架构位置：
#   用户音频 → Qwen3TTSTokenizer.encode() → 离散码
#         → Qwen3TTSForConditionalGeneration（核心 LLM）→ 生成的离散码
#         → Qwen3TTSTokenizer.decode() → 音频波形
# ====================================================================
import base64
import io
import urllib.request
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
# pad_sequence：把不等长的张量列表按 batch_first 维度 padding 成等长批量张量。
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
# AutoFeatureExtractor：HF 的"特征提取器"自动注册机制，根据配置找到对应类，
#                        用于把原始波形转成模型输入格式（如归一化、padding_mask）。

from ..core import (
    Qwen3TTSTokenizerV1Config,   # 25Hz tokenizer 的配置类
    Qwen3TTSTokenizerV1Model,    # 25Hz tokenizer 的模型类
    Qwen3TTSTokenizerV2Config,   # 12Hz tokenizer 的配置类
    Qwen3TTSTokenizerV2Model,    # 12Hz tokenizer 的模型类
)

# AudioInput：encode/decode 接受的音频输入联合类型。
#   - str           ：wav 路径 或 base64 字符串
#   - np.ndarray    ：1 维 float 波形数组（需配 sr）
#   - List[str] / List[np.ndarray]：批量输入
AudioInput = Union[
    str,  # wav path, or base64 string
    np.ndarray,  # 1-D float array
    List[str],
    List[np.ndarray],
]


# --------------------------------------------------------------------
# Qwen3TTSTokenizer：Qwen3-TTS 语音 tokenizer 高层封装
# --------------------------------------------------------------------
# 用途：把底层 25Hz/12Hz 语音 tokenizer 模型封装成易用接口，
#        统一处理多种音频输入形式（路径/URL/base64/numpy）。
# 主要方法：from_pretrained / encode / decode / load_audio / get_*_sample_rate。
# 注意：numpy 波形输入必须额外传 sr（原始采样率），否则无法重采样。
class Qwen3TTSTokenizer:
    """
    A wrapper for Qwen3 TTS Tokenizer 25Hz/12Hz with HuggingFace-style loading.

    - from_pretrained(): loads speech tokenizer model via AutoModel and feature_extractor via AutoFeatureExtractor.
    - encode(): supports wav path(s), base64 audio string(s), numpy array(s).
    - decode(): accepts either the raw model encode output, or a minimal dict/list-of-dicts.

    Notes:
    - For numpy array input, you must pass `sr` so the audio can be resampled to model sample rate.
    - Returned audio is float32 numpy arrays and the output sample rate.
    """

    # 构造函数：通常不直接 new，而是通过 from_pretrained 创建实例。
    # 初始化所有字段为 None，等 from_pretrained 填充。
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.config = None
        self.device = None

    # ------------------------------------------------------------------
    # from_pretrained：HuggingFace 风格的加载入口
    # ------------------------------------------------------------------
    # 做了 3 件事：
    #   1) 向 HF 注册两种 tokenizer 类型（25Hz/12Hz）的 (Config, Model) 映射；
    #   2) 用 AutoFeatureExtractor 加载特征提取器（波形预处理）；
    #   3) 用 AutoModel 加载模型权重（kwargs 透传），并推断运行设备。
    # 入参：
    #   pretrained_model_name_or_path ：HF repo id 或本地模型目录
    #   **kwargs                      ：透传给 AutoModel.from_pretrained
    #       常用：device_map="cuda:0"、dtype=torch.bfloat16（省显存）、
    #             attn_implementation="eager"（不用 flash-attn 时的回退实现）
    # 返回：装好 model/feature_extractor/config/device 的 Qwen3TTSTokenizer 实例。
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "Qwen3TTSTokenizer":
        """
        Initialize tokenizer with HuggingFace `from_pretrained` style.

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory.
            **kwargs (Any):
                Forwarded to `AutoModel.from_pretrained(...)` directly.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="eager".

        Returns:
            Qwen3TTSTokenizer:
                Initialized instance with `model`, `feature_extractor`, `config`.
        """
        inst = cls()

        # 注册两种 tokenizer 类型到 HF 的 Auto 系列：
        # "qwen3_tts_tokenizer_25hz" → V1 (Config, Model)
        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)
        AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)

        # "qwen3_tts_tokenizer_12hz" → V2 (Config, Model)
        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

        # 加载特征提取器：负责把原始波形转成 model.encode 期望的输入格式。
        inst.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        # 加载模型权重，kwargs（device_map/dtype/...）原样透传。
        inst.model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        inst.config = inst.model.config

        # 推断运行设备：优先用模型自带 device 属性，没有则取首个参数所在设备。
        inst.device = getattr(inst.model, "device", None)
        if inst.device is None:
            # fallback: infer from first parameter device
            try:
                inst.device = next(inst.model.parameters()).device
            except StopIteration:
                inst.device = torch.device("cpu")

        return inst

    # 启发式判断字符串是否为 base64 音频：data:audio 前缀 或 无路径分隔符且较长（>256 字符）。
    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        # Heuristic: no filesystem path separators and long enough.
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    # 判断字符串是否为合法的 http(s) URL。
    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    # 把 base64 音频字符串解码成原始 wav 字节流，支持 "data:audio/wav;base64,xxxx" 形式。
    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        # Accept both "data:audio/wav;base64,...." and raw base64
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    # ------------------------------------------------------------------
    # load_audio：从 wav 路径 / URL / base64 字符串加载音频并重采样到目标采样率
    # ------------------------------------------------------------------
    # 入参：
    #   x         ：wav 路径、URL 或 base64 音频字符串（raw 或 data URL）
    #   target_sr ：目标采样率（Hz，每秒声音点数）
    # 返回：1 维 float32 波形数组（已重采样到 target_sr，多声道会混成单声道）。
    def load_audio(
        self,
        x: str,
        target_sr: int,
    ) -> np.ndarray:
        """
        Load audio from wav path or base64 string, then resample to target_sr.

        Args:
            x (str):
                A wav file path, or a base64 audio string (raw or data URL).
            target_sr (int):
                Target sampling rate.

        Returns:
            np.ndarray:
                1-D float32 waveform at target_sr.
        """
        if self._is_url(x):
            # 1) 下载 URL 音频到内存，再用 soundfile 解码。
            with urllib.request.urlopen(x) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif self._is_probably_base64(x):
            # 2) base64 字符串 → 字节流 → soundfile 解码。
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            # 3) 本地文件路径，用 librosa 加载（不重采样，mono=True 转单声道）。
            audio, sr = librosa.load(x, sr=None, mono=True)

        # 多声道情形按均值混成单声道。
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        # 不一致时重采样到目标采样率。
        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

        return audio.astype(np.float32)

    def _normalize_audio_inputs(
        self,
        audios: AudioInput,
        sr: Optional[int],
    ) -> List[np.ndarray]:
        """
        Normalize all supported input types into a list of 1-D numpy float32 waveforms
        at `self.feature_extractor.sampling_rate`.

        Args:
            audios (AudioInput):
                - str: wav path OR base64 audio string
                - np.ndarray: raw waveform (sr must be provided)
                - list[str] / list[np.ndarray]
            sr (Optional[int]):
                Sampling rate for raw numpy input. Required if input is np.ndarray or list[np.ndarray].

        Returns:
            List[np.ndarray]:
                List of float32 waveforms resampled to model input SR.
        """
        # 目标采样率：feature_extractor 期望的输入采样率。
        target_sr = int(self.feature_extractor.sampling_rate)

        # 单条 str/np.ndarray 也包装成列表统一处理。
        if isinstance(audios, (str, np.ndarray)):
            audios = [audios]

        if len(audios) == 0:
            return []

        if isinstance(audios[0], str):
            # 字符串列表：交给 load_audio 处理路径/URL/base64。
            # wav path list or base64 list
            return [self.load_audio(x, target_sr=target_sr) for x in audios]  # type: ignore[arg-type]

        # numpy 列表：必须提供 sr，否则无法重采样。
        if sr is None:
            raise ValueError("For numpy waveform input, you must provide `sr` (original sampling rate).")

        out: List[np.ndarray] = []
        for a in audios:  # type: ignore[assignment]
            if not isinstance(a, np.ndarray):
                # 不允许混合类型（一半路径一半 numpy）。
                raise TypeError("Mixed input types are not supported. Use all paths/base64 or all numpy arrays.")
            if a.ndim > 1:
                a = np.mean(a, axis=-1)                # 多声道混成单声道
            if int(sr) != target_sr:
                a = librosa.resample(y=a.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
            out.append(a.astype(np.float32))
        return out

    # ------------------------------------------------------------------
    # encode：批量把音频编码成离散码（codec tokens）
    # ------------------------------------------------------------------
    # 用途：把波形喂给语音 tokenizer 模型，得到离散码序列（及可选条件）。
    # 入参：
    #   audios      ：音频输入，支持 numpy 波形（需 sr）/ str（路径/base64）/ 列表
    #   sr          ：numpy 波形的原始采样率（str 输入时无需）
    #   return_dict ：True=返回 ModelOutput 对象；False=返回原始 tuple
    # 返回：
    #   - 25Hz (V1)：含 audio_codes (List[(codes_len,)])、xvectors (说话人向量)、
    #                ref_mels (参考梅尔频谱，decode 时作为条件)
    #   - 12Hz (V2)：仅含 audio_codes (List[(codes_len, num_quantizers)])，
    #                多量化器残差码
    def encode(
        self,
        audios: AudioInput,
        sr: Optional[int] = None,
        return_dict: bool = True,
    ):
        """
        Batch-encode audio into discrete codes (and optional conditioning, depending on 25Hz/12Hz).

        Args:
            audios (AudioInput):
                Supported forms:
                - np.ndarray: waveform (requires sr)
                - list[np.ndarray]: waveforms (requires sr)
                - str: wav path OR base64 audio string
                - list[str]: wav paths and/or base64 strings
            sr (Optional[int], default=None):
                Original sampling rate for numpy waveform input.
            return_dict (bool, default=True):
                Forwarded to model.encode(...). If True, returns ModelOutput.

        Returns:
            25Hz:
                Qwen3TTSTokenizerV1EncoderOutput (if return_dict=True) with fields:
                  - audio_codes: List[torch.LongTensor] each (codes_len,)
                  - xvectors:   List[torch.FloatTensor] each (xvector_dim,)
                  - ref_mels:   List[torch.FloatTensor] each (mel_len, mel_dim)
            12Hz:
                Qwen3TTSTokenizerV2EncoderOutput (if return_dict=True) with fields:
                  - audio_codes: List[torch.LongTensor] each (codes_len, num_quantizers)

            If return_dict=False, returns the raw tuple from model.encode.
        """
        # 先把多种形式统一成目标采样率下的波形列表。
        wavs = self._normalize_audio_inputs(audios, sr=sr)

        # 用特征提取器把波形转成模型输入（input_values + padding_mask）。
        inputs = self.feature_extractor(
            raw_audio=wavs,
            sampling_rate=int(self.feature_extractor.sampling_rate),
            return_tensors="pt",
        )
        # 搬到模型所在设备和数据类型。
        inputs = inputs.to(self.device).to(self.model.dtype)

        with torch.inference_mode():
            # model.encode 期望 (B, T) 形状的波形和 padding_mask，
            # feature_extractor 可能多加了一维通道维，这里 squeeze(1) 去掉。
            # model.encode expects (B, T) and (B, T)
            enc = self.model.encode(
                inputs["input_values"].squeeze(1),
                inputs["padding_mask"].squeeze(1),
                return_dict=return_dict,
            )
        return enc

    # ------------------------------------------------------------------
    # decode：把离散码（codec tokens）解码回音频波形
    # ------------------------------------------------------------------
    # 用途：把 encode 输出（或自定义的 dict/list[dict]）解码成可听的波形。
    # 支持 3 种输入形式：
    #   1) 直接传 encode() 的原始 ModelOutput（推荐）；
    #   2) 传 dict（最小形式）：25Hz 需 {"audio_codes","xvectors","ref_mels"}，
    #                            12Hz 只需 {"audio_codes"}；
    #   3) 传 list[dict]：每条样本一个 dict，适合自定义 pipeline。
    #   值可以是 torch.Tensor 或 numpy 数组，内部会自动转换。
    # 入参：
    #   encoded ：见上 3 种形式
    # 返回：(wavs: List[np.ndarray 1维 float32], sample_rate: int)
    def decode(
        self,
        encoded,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Decode back to waveform.

        Usage:
        1) Pass the raw output of `encode(...)` directly (recommended).
           - 25Hz: expects fields audio_codes, xvectors, ref_mels
           - 12Hz: expects field audio_codes
        2) Pass a dict or list[dict] (minimal form) for custom pipelines:
           - 25Hz dict keys: {"audio_codes", "xvectors", "ref_mels"}
           - 12Hz dict keys: {"audio_codes"}
           Values can be torch tensors or numpy arrays.

        Args:
            encoded (Any):
                - ModelOutput returned by `encode()`, OR
                - dict, OR
                - list[dict]

        Returns:
            Tuple[List[np.ndarray], int]:
                - wavs: list of 1-D float32 numpy arrays
                - sample_rate: int, model output sampling rate
        """
        # 先拿到当前 tokenizer 类型，决定 decode 的调用签名。
        model_type = self.model.get_model_type()

        # 小工具：把 numpy/标量转成 torch.Tensor，可指定 dtype。
        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return x
            x = np.asarray(x)
            t = torch.from_numpy(x)
            if dtype is not None:
                t = t.to(dtype)
            return t

        # 统一从三种形式里抽出 audio_codes / xvectors / ref_mels 三个字段。
        # Normalize `encoded` into the same shapes as the official demo uses.
        if hasattr(encoded, "audio_codes"):
            # ModelOutput from encode()
            audio_codes_list = encoded.audio_codes
            xvectors_list = getattr(encoded, "xvectors", None)
            ref_mels_list = getattr(encoded, "ref_mels", None)
        elif isinstance(encoded, dict):
            # 单个 dict
            audio_codes_list = encoded["audio_codes"]
            xvectors_list = encoded.get("xvectors", None)
            ref_mels_list = encoded.get("ref_mels", None)
        elif isinstance(encoded, list):
            # list of dicts：逐条抽出字段，组成列表。
            # list of dicts
            audio_codes_list = [e["audio_codes"] for e in encoded]
            xvectors_list = [e["xvectors"] for e in encoded] if ("xvectors" in encoded[0]) else None
            ref_mels_list = [e["ref_mels"] for e in encoded] if ("ref_mels" in encoded[0]) else None
        else:
            raise TypeError("`encoded` must be an encode output, a dict, or a list of dicts.")

        # 把 audio_codes 整理成 (B, C) 或 (B, C, Q) 的批量张量。
        # Ensure list form for per-sample tensors
        if isinstance(audio_codes_list, torch.Tensor):
            # 单个张量：补一个 batch 维。
            # Could be a single sample tensor or an already padded batch tensor.
            t = audio_codes_list
            if t.dim() == 1:
                # 25Hz 单样本：(C,) -> (1, C)
                # 25Hz single sample: (C,) -> (1, C)
                t = t.unsqueeze(0)
            elif t.dim() == 2:
                # 12Hz 单样本：(C, Q) -> (1, C, Q)
                # 12Hz single sample: (C, Q) -> (1, C, Q)
                t = t.unsqueeze(0)
            audio_codes_padded = t.to(self.device)
        else:
            # 列表形式：逐条转 long 张量，再用 pad_sequence padding 成等长批量。
            # List[Tensor/np]
            audio_codes_list = [_to_tensor(c, dtype=torch.long) for c in audio_codes_list]
            audio_codes_padded = pad_sequence(audio_codes_list, batch_first=True, padding_value=-1).to(self.device)

        with torch.inference_mode():
            if model_type == "qwen3_tts_tokenizer_25hz":
                # 25Hz decode 需要额外传说话人向量 + 参考梅尔频谱作为条件。
                if xvectors_list is None or ref_mels_list is None:
                    raise ValueError("25Hz decode requires `xvectors` and `ref_mels`.")

                # 整理 xvectors → (B, D) 批量张量。
                if isinstance(xvectors_list, torch.Tensor):
                    xvectors_batch = xvectors_list
                    if xvectors_batch.dim() == 1:  # (D,) -> (1, D)
                        xvectors_batch = xvectors_batch.unsqueeze(0)
                    xvectors_batch = xvectors_batch.to(self.device).to(self.model.dtype)
                else:
                    xvectors_list = [_to_tensor(x, dtype=torch.float32) for x in xvectors_list]
                    xvectors_batch = torch.stack(xvectors_list, dim=0).to(self.device).to(self.model.dtype)

                # 整理 ref_mels → (B, T, M) 批量张量（不等长用 pad_sequence padding）。
                if isinstance(ref_mels_list, torch.Tensor):
                    ref_mels_padded = ref_mels_list
                    if ref_mels_padded.dim() == 2:  # (T, M) -> (1, T, M)
                        ref_mels_padded = ref_mels_padded.unsqueeze(0)
                    ref_mels_padded = ref_mels_padded.to(self.device).to(self.model.dtype)
                else:
                    ref_mels_list = [_to_tensor(m, dtype=torch.float32) for m in ref_mels_list]
                    ref_mels_padded = pad_sequence(ref_mels_list, batch_first=True, padding_value=0).to(self.device).to(self.model.dtype)

                # 调用底层 25Hz decode，返回 audio_values（波形张量）。
                dec = self.model.decode(audio_codes_padded, xvectors_batch, ref_mels_padded, return_dict=True)
                wav_tensors = dec.audio_values

            elif model_type == "qwen3_tts_tokenizer_12hz":
                # 12Hz decode 只需 audio_codes（多量化器残差码自带全部信息）。
                dec = self.model.decode(audio_codes_padded, return_dict=True)
                wav_tensors = dec.audio_values

            else:
                raise ValueError(f"Unknown model type: {model_type}")

        # 张量 → float32 numpy 数组，逐条返回。
        wavs = [w.to(torch.float32).detach().cpu().numpy() for w in wav_tensors]
        return wavs, int(self.model.get_output_sample_rate())

    # 获取底层 tokenizer 模型类型字符串（"qwen3_tts_tokenizer_25hz" / "12hz"）。
    def get_model_type(self) -> str:
        """
        Get the underlying tokenizer model type.

        Returns:
            str: Model type string from `self.model.config.model_type`
                (e.g. "qwen3_tts_tokenizer_25hz" / "qwen3_tts_tokenizer_12hz").
        """
        return self.model.get_model_type()

    # 获取 encode 期望的输入采样率（Hz）。
    def get_input_sample_rate(self) -> int:
        """
        Get the expected input sample rate for encoding.

        Returns:
            int: Input sample rate (Hz).
        """
        return int(self.model.get_input_sample_rate())

    # 获取 decode 输出波形的采样率（Hz）。
    def get_output_sample_rate(self) -> int:
        """
        Get the output sample rate for decoded waveforms.

        Returns:
            int: Output sample rate (Hz).
        """
        return int(self.model.get_output_sample_rate())

    # 获取 encode 的下采样率：每个码对应多少个波形采样点。
    def get_encode_downsample_rate(self) -> int:
        """
        Get the encoder downsample rate (waveform samples per code step).

        Returns:
            int: Encode downsample rate.
        """
        return int(self.model.get_encode_downsample_rate())

    # 获取 decode 的上采样率：每个码对应多少个波形采样点。
    def get_decode_upsample_rate(self) -> int:
        """
        Get the decoder upsample rate (waveform samples per code step).

        Returns:
            int: Decode upsample rate.
        """
        return int(self.model.get_decode_upsample_rate())