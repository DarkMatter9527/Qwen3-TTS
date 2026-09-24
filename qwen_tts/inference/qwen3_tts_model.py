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
# 文件总览：qwen3_tts_model.py
# --------------------------------------------------------------------
# 本文件是 Qwen3-TTS 端到端 TTS 大模型的"高层 API"封装层。
# 它把底层核心模型（core/models.py 里的 Qwen3TTSForConditionalGeneration）
# 和文本 processor（Qwen3TTSProcessor）串成一个面向用户的"好用接口"。
#
# 主要导出：
#   - VoiceClonePromptItem  ：声音克隆提示项的数据容器（dataclass）。
#   - Qwen3TTSModel         ：高层 API 类，用户通过它来加载模型并生成语音。
#
# Qwen3TTSModel 主要方法：
#   - from_pretrained(path, **kwargs)
#         HuggingFace 风格的模型加载入口，注册 qwen3_tts 类型并加载模型/processor。
#   - create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode)
#         （仅 base 模型）根据参考音频和参考文字，构造声音克隆提示项列表。
#   - generate_voice_clone(text, language, ref_audio, ref_text, ...)
#         （仅 base 模型）声音克隆生成，支持 ICL 与 x_vector_only 两种模式。
#   - generate_voice_design(text, instruct, language, ...)
#         （仅 voice_design 模型）按自然语言描述生成自定义音色语音。
#   - generate_custom_voice(text, speaker, language, instruct, ...)
#         （仅 custom_voice 模型）使用预置音色（Vivian/Ryan/...）合成语音。
#   - get_supported_speakers / get_supported_languages
#         查询当前模型支持的说话人 / 语言列表。
#
# 整体架构位置：
#   用户 → Qwen3TTSModel（本文件，高层 API）
#        → Qwen3TTSForConditionalGeneration.generate()（底层核心模型）
#        → speech_tokenizer.decode()（语音 tokenizer，见 qwen3_tts_tokenizer.py）
#        → np.ndarray 音频波形 + 采样率
#
# 统一返回格式：Tuple[List[np.ndarray], int]，即 (每条音频波形列表, 采样率)。
# ====================================================================
import base64
import io
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor
# AutoConfig/AutoModel/AutoProcessor：HuggingFace 的自动注册机制，根据 model_type
# 字符串自动找到对应的 Config / Model / Processor 类，免去手动指定。

from ..core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

# AudioLike：表示"一段音频输入"的联合类型，接受三种形式：
#   - str                     ：wav 文件路径 / URL / base64 字符串
#   - np.ndarray              ：原始波形数组（必须额外提供采样率 sr）
#   - Tuple[np.ndarray, int]  ：(波形, 采样率) 元组
AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

# MaybeList：可以是单个对象，也可以是该对象的列表，方便统一处理标量/批量输入。
MaybeList = Union[Any, List[Any]]


# --------------------------------------------------------------------
# VoiceClonePromptItem：声音克隆"提示项"数据容器
# --------------------------------------------------------------------
# 用途：保存一条声音克隆样本所需的所有信息，便于传给底层模型的 generate()。
# 字段与 Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)
# 的入参一一对应。
# 通俗理解：声音克隆就是"给一段参考音频让模型模仿"，这个对象就是装参考信息的盒子。
@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """
    # ref_code：参考音频被语音 tokenizer 编码后得到的"离散码"序列。
    #   - 25Hz tokenizer：形状 (T,) 一维
    #   - 12Hz tokenizer：形状 (T, Q) 二维，Q 是量化器层数
    # x_vector_only 模式下置为 None（不使用参考码）。
    ref_code: Optional[torch.Tensor]                 # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    # ref_spk_embedding：参考音频提取出的"说话人向量"（x-vector / speaker embedding），
    # 即把"是谁的声音"浓缩成一个定长向量，形状 (D,)。
    ref_spk_embedding: torch.Tensor                  # (D,)
    # x_vector_only_mode：True=只用说话人向量克隆音色（无需参考文字）；
    # False=走 ICL 模式（同时用参考码 + 参考文字）。
    x_vector_only_mode: bool
    # icl_mode：是否启用 ICL（In-Context Learning，上下文学习）。
    # ICL 即把"参考音频 + 对应文字"当作例题喂给模型，让它模仿该说话人。
    icl_mode: bool
    # ref_text：参考音频对应的文字稿，ICL 模式下必填，x_vector_only 模式下可为 None。
    ref_text: Optional[str] = None


# --------------------------------------------------------------------
# Qwen3TTSModel：Qwen3-TTS 的高层 API 主类
# --------------------------------------------------------------------
# 用途：把"加载模型 + 文本处理 + 调用底层 generate + 解码回音频"串成一个易用接口。
# 支持三种模型类型（由底层 model.tts_model_type 决定）：
#   - "base"         ：声音克隆模型（generate_voice_clone / create_voice_clone_prompt）
#   - "custom_voice" ：预置音色模型（generate_custom_voice）
#   - "voice_design" ：自然语言描述音色模型（generate_voice_design）
# 所有生成方法统一返回 (wavs: List[np.ndarray], sample_rate: int)。
class Qwen3TTSModel:
    """
    A HuggingFace-style wrapper for Qwen3 TTS models (CustomVoice/VoiceDesign/Base) that provides:
      - from_pretrained() initialization via AutoModel/AutoProcessor
      - generation APIs for:
          * CustomVoice: generate_custom_voice()
          * VoiceDesign: generate_voice_design()
          * Base: generate_voice_clone() + create_voice_clone_prompt()
      - consistent output: (wavs: List[np.ndarray], sample_rate: int)

    Notes:
      - This wrapper expects the underlying model class to be `Qwen3TTSForConditionalGeneration`
      - Language / speaker validation is done via model methods:
          model.get_supported_languages(), model.get_supported_speakers()
    """

    # 构造函数：通常不直接 new，而是通过 from_pretrained 创建实例。
    # 入参：
    #   model             ：已加载好的底层 Qwen3TTSForConditionalGeneration 模型
    #   processor         ：对应的 Qwen3TTSProcessor，负责文本 tokenization
    #   generate_defaults ：来自 generate_config.json 的默认采样参数（可被调用方覆盖）
    def __init__(self, model: Qwen3TTSForConditionalGeneration, processor, generate_defaults: Optional[Dict[str, Any]] = None):
        self.model = model
        self.processor = processor
        self.generate_defaults = generate_defaults or {}

        # 推断运行设备（CPU/GPU），优先用模型自带的 device 属性，没有则取首个参数所在设备。
        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # from_pretrained：HuggingFace 风格的模型加载入口（推荐用户使用）
    # ------------------------------------------------------------------
    # 做了 4 件事：
    #   1) 向 HF 注册 "qwen3_tts" 类型 → (Config, Model, Processor) 的映射关系；
    #   2) 用 AutoModel.from_pretrained 加载模型权重（kwargs 原样透传）；
    #   3) 用 AutoProcessor.from_pretrained 加载文本 processor；
    #   4) 读取 generate_config.json 里的默认采样参数。
    # 入参：
    #   pretrained_model_name_or_path ：HF repo id 或本地模型目录
    #   **kwargs                      ：透传给 AutoModel.from_pretrained
    #       常用：device_map="cuda:0"、dtype=torch.bfloat16（省显存）、
    #             attn_implementation="flash_attention_2"（加速注意力）
    # 返回：装好 model/processor/默认参数的 Qwen3TTSModel 实例。
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "Qwen3TTSModel":
        """
        Load a Qwen3 TTS model and its processor in HuggingFace `from_pretrained` style.

        This method:
          1) Loads config via AutoConfig (so your side can register model_type -> config/model).
          2) Loads the model via AutoModel.from_pretrained(...), forwarding `kwargs` unchanged.
          3) Loads the processor via AutoProcessor.from_pretrained(model_path).
          4) Loads optional `generate_config.json` from the model directory/repo snapshot if present.

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory of the model.
            **kwargs:
                Forwarded as-is into `AutoModel.from_pretrained(...)`.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2".

        Returns:
            Qwen3TTSModel:
                Wrapper instance containing `model`, `processor`, and generation defaults.
        """
        # 把 "qwen3_tts" 这个 model_type 字符串注册到 HF 的 Auto 系列，
        # 之后 AutoModel/Processor 看到 qwen3_tts 配置就能自动找到对应类。
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        # 加载模型权重，kwargs（device_map/dtype/...）原样透传。
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # 类型校验：确保加载到的确实是 Qwen3-TTS 主模型，否则后续调用会出错。
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration. "
            )

        # 加载文本 processor；fix_mistral_regex=True 用于修正某些 tokenizer 的正则问题。
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True,)

        # 读取模型自带的 generate_config（默认采样参数），供后续 _merge_generate_kwargs 使用。
        generate_defaults = model.generate_config
        return cls(model=model, processor=processor, generate_defaults=generate_defaults)

    # ------------------------------------------------------------------
    # 以下为内部辅助方法（私有，以 _ 开头）
    # ------------------------------------------------------------------

    # 获取模型支持的语言集合（小写化），用于校验。模型未实现该方法则返回 None 表示不限制。
    def _supported_languages_set(self) -> Optional[set]:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    # 获取模型支持的预置说话人集合（小写化），用于校验。模型未实现该方法则返回 None。
    def _supported_speakers_set(self) -> Optional[set]:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _validate_languages(self, languages: List[str]) -> None:
        """
        Validate that requested languages are supported by the model.

        Args:
            languages (List[str]): Language names for each sample.

        Raises:
            ValueError: If any language is not supported.
        """
        supported = self._supported_languages_set()
        # 模型不限制语言时直接放行。
        if supported is None:
            return

        bad = []
        for lang in languages:
            # None 视为非法（调用方没显式指定语言时应传 "Auto" 而非 None）。
            if lang is None:
                bad.append(lang)
                continue
            if str(lang).lower() not in supported:
                bad.append(lang)
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")

    def _validate_speakers(self, speakers: List[Optional[str]]) -> None:
        """
        Validate that requested speakers are supported by the Instruct model.

        Args:
            speakers (List[Optional[str]]): Speaker names for each sample.

        Raises:
            ValueError: If any speaker is not supported.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return

        bad = []
        for spk in speakers:
            # None 或空串视为"不指定 instruct"（不校验），与 generate 流程一致。
            if spk is None or spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(spk)
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")

    # 启发式判断字符串是否为 base64 音频：data:audio 前缀 或 无路径分隔符且较长（>256 字符）。
    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
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
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    # 根据字符串加载音频：URL / base64 / 本地路径，统一返回 (float32 波形, 原始采样率)。
    # 多声道会被按均值混合为单声道。
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
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

        # 双声道等多声道情形按均值混成单声道。
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
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
        # 统一成 list 处理，单条也包装成单元素 list。
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                # 字符串：交给 _load_audio_to_np 处理 URL/base64/路径。
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                # (波形, 采样率) 元组：直接取用，并强转 float32。
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                # 裸 numpy 波形无法获知采样率，拒绝并提示用户改用元组形式。
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        # 二次保险：把任何残留的多声道数组混成单声道。
        for i, a in enumerate(out):
            if a[0].ndim > 1:
                a[0] = np.mean(a[0], axis=-1).astype(np.float32)
                out[i] = (a[0], a[1])
        return out

    # 工具方法：把单个对象或列表统一成列表（标量 → [标量]）。
    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    # 构造 assistant 段落文本：用 Qwen 对话模板包裹待合成文字，
    # 并再加一个空的 assistant 起始符，引导模型在此处开始生成 codec 码。
    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    # 构造参考文字（ref_text）的对话段落，用于 ICL 模式下作为"例题"喂给模型。
    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    # 构造 instruct（自然语言音色描述）的 user 段落，用于 voice_design / custom_voice。
    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    # 用 processor 把多条文本批量 tokenize 成 input_ids（已搬到 self.device）。
    # 每条结果至少是 2 维 (batch=1, seq)，方便后续批量调用。
    def _tokenize_texts(self, texts: List[str]) -> List[torch.Tensor]:
        input_ids = []
        for text in texts:
            input = self.processor(text=text, return_tensors="pt", padding=True)
            input_id = input["input_ids"].to(self.device)
            # 1 维时补一个 batch 维，保证下游统一为 (1, seq) 形状。
            input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
            input_ids.append(input_id)
        return input_ids

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Merge user-provided generation arguments with defaults from `generate_config.json`.

        Rule:
          - If the user explicitly passes a value (not None), use it.
          - Otherwise, use the value from generate_config.json if present.
          - Otherwise, fall back to the hard defaults.

        Args:
            do_sample, top_k, top_p, temperature, repetition_penalty,
            subtalker_dosample, subtalker_top_k, subtalker_top_p, subtalker_temperature, max_new_tokens:
                Common generation parameters.
            **kwargs:
                Other arguments forwarded to model.generate().

        Returns:
            Dict[str, Any]: Final kwargs to pass into model.generate().
        """
        # 采样参数术语通俗解释：
        #   do_sample=True        ：开启采样（而非贪心解码），让输出有随机性、更自然。
        #   top_k=50               ：每步只从概率最高的 50 个候选 token 中采样。
        #   top_p=1.0              ：核采样阈值，1.0 表示不裁剪（保留全部候选累积概率）。
        #   temperature=0.9        ：温度，越高越随机，越低越确定。
        #   repetition_penalty    ：对已出现 token 的概率做惩罚，抑制重复。
        #   max_new_tokens=2048   ：最多生成多少个 codec token。
        #   subtalker_*           ：12Hz tokenizer 的"子说话人"模型专属采样参数
        #                            （qwen3-tts-tokenizer-v2 才用到）。
        hard_defaults = dict(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=2048,
        )

        # 三级回退：用户显式传入 > generate_config.json 默认 > 硬编码默认。
        def pick(name: str, user_val: Any) -> Any:
            if user_val is not None:
                return user_val
            if name in self.generate_defaults:
                return self.generate_defaults[name]
            return hard_defaults[name]

        # 先放进其它 kwargs，再用解析后的采样参数覆盖，得到最终 generate() 入参。
        merged = dict(kwargs)
        merged.update(
            do_sample=pick("do_sample", do_sample),
            top_k=pick("top_k", top_k),
            top_p=pick("top_p", top_p),
            temperature=pick("temperature", temperature),
            repetition_penalty=pick("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
            max_new_tokens=pick("max_new_tokens", max_new_tokens),
        )
        return merged

    # voice clone model
    # ------------------------------------------------------------------
    # create_voice_clone_prompt：根据参考音频（+ 可选参考文字）构造声音克隆提示
    # ------------------------------------------------------------------
    # 用途：把"参考音频"转成模型 generate() 需要的 (ref_code, ref_spk_embedding) 等字段。
    # 两种克隆模式：
    #   - x_vector_only_mode=True ：只用说话人向量克隆音色，无需参考文字。
    #   - x_vector_only_mode=False：走 ICL（上下文学习）模式，参考文字必填，
    #                                模型会基于"参考码 + 参考文字"模仿说话人。
    # 流程：
    #   1) 校验必须是 base 模型；
    #   2) 把标量入参广播成与 ref_audio 等长的列表；
    #   3) 用 speech_tokenizer.encode 把参考音频编码成离散码 ref_code；
    #   4) 把音频重采样到说话人编码器所需的采样率（通常 24kHz），
    #      用 extract_speaker_embedding 提取说话人向量；
    #   5) 组装成 VoiceClonePromptItem 列表返回。
    # 入参：
    #   ref_audio           ：参考音频，支持 str(路径/URL/base64) / (np.ndarray, sr) / 列表
    #   ref_text            ：参考文字，ICL 模式下必填；标量会广播到每条音频
    #   x_vector_only_mode  ：是否只用说话人向量；标量会广播
    # 返回：List[VoiceClonePromptItem]，每条样本一个提示项。
    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: Union[AudioLike, List[AudioLike]],
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
    ) -> List[VoiceClonePromptItem]:
        """
        Build voice-clone prompt items from reference audio (and optionally reference text) using Base model.

        Modes:
          - x_vector_only_mode=True:
              Only speaker embedding is used to clone voice; ref_text/ref_code are ignored.
              This is mutually exclusive with ICL.
          - x_vector_only_mode=False:
              ICL mode is enabled automatically (icl_mode=True). In this case ref_text is required,
              because the model continues/conditions on the reference text + reference speech codes.

        Batch behavior:
          - ref_audio can be a single item or a list.
          - ref_text and x_vector_only_mode can be scalars or lists.
          - If any of them are lists with length > 1, lengths must match.

        Audio input:
          - str: local wav path / URL / base64
          - (np.ndarray, sr): waveform + sampling rate

        Args:
            ref_audio:
                Reference audio(s) used to extract:
                  - ref_code via `model.speech_tokenizer.encode(...)`
                  - ref_spk_embedding via `model.extract_speaker_embedding(...)` (resampled to 24k)
            ref_text:
                Reference transcript(s). Required when x_vector_only_mode=False (ICL mode).
            x_vector_only_mode:
                Whether to use speaker embedding only. If False, ICL mode will be used.

        Returns:
            List[VoiceClonePromptItem]:
                List of prompt items that can be converted into `voice_clone_prompt` dict.

        Raises:
            ValueError:
                - If x_vector_only_mode=False but ref_text is missing.
                - If batch lengths mismatch.
        """
        # 仅 base 类型模型支持声音克隆，其它类型直接报错并提示看 Model Card。
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support create_voice_clone_prompt, Please check Model Card or Readme for more details."
            )

        # 把标量广播成与 ref_audio 等长：ref_text、x_vector_only_mode 若是标量则复制 N 份。
        ref_audio_list = self._ensure_list(ref_audio)
        ref_text_list = self._ensure_list(ref_text) if isinstance(ref_text, list) else ([ref_text] * len(ref_audio_list))
        xvec_list = self._ensure_list(x_vector_only_mode) if isinstance(x_vector_only_mode, list) else ([x_vector_only_mode] * len(ref_audio_list))

        # 长度对齐校验。
        if len(ref_text_list) != len(ref_audio_list) or len(xvec_list) != len(ref_audio_list):
            raise ValueError(
                f"Batch size mismatch: ref_audio={len(ref_audio_list)}, ref_text={len(ref_text_list)}, x_vector_only_mode={len(xvec_list)}"
            )

        # 把多种形式的音频输入统一成 (波形, 采样率) 列表。
        normalized = self._normalize_audio_inputs(ref_audio_list)

        # 分别收集波形和采样率，便于后续批量编码。
        ref_wavs_for_code: List[np.ndarray] = []
        ref_sr_for_code: List[int] = []
        for wav, sr in normalized:
            ref_wavs_for_code.append(wav)
            ref_sr_for_code.append(sr)

        # 编码成离散码：所有样本采样率一致时一次性批量编码，否则逐条编码。
        # 这里调用的就是 qwen3_tts_tokenizer.py 里的 Qwen3TTSTokenizer.encode()。
        if len(set(ref_sr_for_code)) == 1:
            enc = self.model.speech_tokenizer.encode(ref_wavs_for_code, sr=ref_sr_for_code[0])
            ref_codes = enc.audio_codes
        else:
            ref_codes = []
            for wav, sr in normalized:
                ref_codes.append(self.model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0])

        # 逐条组装提示项：提取说话人向量 + 决定是否使用 ICL。
        items: List[VoiceClonePromptItem] = []
        for i, ((wav, sr), code, rtext, xvec_only) in enumerate(zip(normalized, ref_codes, ref_text_list, xvec_list)):
            # ICL 模式下必须提供参考文字，否则无法做"例题"。
            if not xvec_only:
                if rtext is None or rtext == "":
                    raise ValueError(f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}")

            # 说话人编码器有固定输入采样率（speaker_encoder_sample_rate，通常 24kHz），
            # 不一致时先重采样到该采样率，否则提取出的向量质量会下降。
            wav_resample = wav
            if sr != self.model.speaker_encoder_sample_rate:
                wav_resample = librosa.resample(y=wav_resample.astype(np.float32),
                                           orig_sr=int(sr),
                                           target_sr=self.model.speaker_encoder_sample_rate)

            # 提取说话人向量（x-vector / speaker embedding）。
            spk_emb = self.model.extract_speaker_embedding(audio=wav_resample,
                                                           sr=self.model.speaker_encoder_sample_rate)

            items.append(
                VoiceClonePromptItem(
                    ref_code=None if xvec_only else code,  # x_vector_only 模式不用参考码
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=bool(xvec_only),
                    icl_mode=bool(not xvec_only),            # 二选一：非 x_vector_only 即 ICL
                    ref_text=rtext,
                )
            )
        return items

    # 把 VoiceClonePromptItem 列表展平成底层 generate() 期望的 dict 形式：
    #   {"ref_code": [...], "ref_spk_embedding": [...], "x_vector_only_mode": [...], "icl_mode": [...]}
    def _prompt_items_to_voice_clone_prompt(self, items: List[VoiceClonePromptItem]) -> Dict[str, Any]:
        return dict(
            ref_code=[it.ref_code for it in items],
            ref_spk_embedding=[it.ref_spk_embedding for it in items],
            x_vector_only_mode=[it.x_vector_only_mode for it in items],
            icl_mode=[it.icl_mode for it in items],
        )

    # voice clone model
    # ------------------------------------------------------------------
    # generate_voice_clone：声音克隆合成语音（仅 base 模型）
    # ------------------------------------------------------------------
    # 用途：给定待合成文字 + 参考音频（或预生成的克隆提示），合成模仿参考音色的语音。
    # 提示来源三选一：
    #   - 现传 (ref_audio, ref_text, x_vector_only_mode)：本方法内部调 create_voice_clone_prompt 构建；
    #   - voice_clone_prompt 为 List[VoiceClonePromptItem]：直接用；
    #   - voice_clone_prompt 为 dict：直接用（已展平形式）。
    # 关键参数：
    #   text               ：待合成文字（标量或列表，列表表示批量）
    #   language           ：每条样本语言，None→"Auto" 自动识别；标量会广播
    #   ref_audio/ref_text/x_vector_only_mode：见 create_voice_clone_prompt
    #   voice_clone_prompt ：预生成的克隆提示，避免重复编码
    #   non_streaming_mode  ：True=一次给全部文字；False=模拟流式文字输入（双轨混合流式）
    #   **kwargs            ：采样参数（do_sample/top_k/top_p/temperature/repetition_penalty/
    #                         max_new_tokens/subtalker_*），详见 _merge_generate_kwargs
    # 返回：(wavs: List[np.ndarray], sample_rate: int)
    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Voice clone speech using the Base model.

        You can provide either:
          - (ref_audio, ref_text, x_vector_only_mode) and let this method build the prompt, OR
          - `VoiceClonePromptItem` returned by `create_voice_clone_prompt`, OR
          - a list of `VoiceClonePromptItem` returned by `create_voice_clone_prompt`.

        `ref_audio` Supported forms:
        - str: wav path / URL / base64 audio string
        - (np.ndarray, sr): waveform + sampling rate
        - list of the above

        Input flexibility:
          - text/language can be scalar or list.
          - prompt can be single or batch.
          - If batch mode (len(text)>1), lengths must match.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            ref_audio:
                Reference audio(s) for prompt building. Required if voice_clone_prompt is not provided.
            ref_text:
                Reference text(s) used for ICL mode (required when x_vector_only_mode=False).
            x_vector_only_mode:
                If True, only speaker embedding is used (ignores ref_text/ref_code).
                If False, ICL mode is used automatically.
            voice_clone_prompt:
                list[VoiceClonePromptItem] from `create_voice_clone_prompt`.
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`,
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If batch sizes mismatch or required prompt inputs are missing.
        """
        # 仅 base 模型支持。
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_clone, Please check Model Card or Readme for more details."
            )

        # 文本和语言统一成列表；language 缺省为 "Auto"（自动识别）。
        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        # 单条语言广播到 N 条样本，便于后续对齐校验。
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(texts) != len(languages):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}")

        # 校验语言是否被模型支持。
        self._validate_languages(languages)

        # 三种提示来源分支：
        if voice_clone_prompt is None:
            # 分支 A：没有预生成提示，必须传 ref_audio，内部现建提示。
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            prompt_items = self.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
            # 单条提示广播到多条文本（同一音色合成多段）。
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            # 保存 ref_text，后面要用来构造 ref_ids（ICL 模式的"例题"文字 token）。
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            # 分支 B/C：用户已提供提示。
            if isinstance(voice_clone_prompt, list):
                # 分支 B：List[VoiceClonePromptItem]。
                prompt_items = voice_clone_prompt
                if len(prompt_items) == 1 and len(texts) > 1:
                    prompt_items = prompt_items * len(texts)
                if len(prompt_items) != len(texts):
                    raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                ref_texts_for_ids = [it.ref_text for it in prompt_items]
            else:
                # 分支 C：已经是 dict 形式，直接用，无法重建 ref_ids。
                voice_clone_prompt_dict = voice_clone_prompt
                ref_texts_for_ids = None

        # 构造待合成文字的 input_ids：用 assistant 模板包裹，引导模型在此处生成 codec。
        input_texts = [self._build_assistant_text(t) for t in texts]
        input_ids = self._tokenize_texts(input_texts)

        # 构造参考文字的 ref_ids（仅 ICL 模式），作为"例题"喂给模型。
        ref_ids = None
        if ref_texts_for_ids is not None:
            ref_ids = []
            for i, rt in enumerate(ref_texts_for_ids):
                if rt is None or rt == "":
                    ref_ids.append(None)          # x_vector_only 模式无参考文字
                else:
                    ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                    ref_ids.append(ref_tok)

        # 合并采样参数（用户传入 > generate_config.json > 硬编码默认）。
        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        # 调用底层核心模型生成 codec 码序列。
        # non_streaming_mode=False 时模拟流式文字输入（双轨混合流式），适合长文本。
        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        # ICL 模式下，生成的 codec 前面要拼上参考码，才能正确解码出完整音频
        # （因为模型生成时是基于"参考码继续写"的）。
        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        # 解码 codec → 波形：调用语音 tokenizer 的 decode()。
        wavs_all, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])

        # ICL 模式下，解码出的波形包含"参考音频段"，需要按比例切掉前段，只保留新生成部分。
        wavs_out: List[np.ndarray] = []
        for i, wav in enumerate(wavs_all):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                # 按码长度比例估算参考音频在波形里占多长，切掉前段。
                ref_len = int(ref_code_list[i].shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        return wavs_out, fs

    # voice design model
    # ------------------------------------------------------------------
    # generate_voice_design：按自然语言描述生成自定义音色语音（仅 voice_design 模型）
    # ------------------------------------------------------------------
    # 用途：用户用一段自然语言（instruct）描述想要的音色/风格，模型据此合成语音。
    #   例：instruct="一个温柔的年轻女声，语速偏慢"。
    # 与 generate_custom_voice 的区别：本方法不需要指定具体 speaker 名字，
    # 而是让模型根据描述自由生成音色。
    # 关键参数：
    #   text               ：待合成文字（标量或批量）
    #   instruct           ：音色/风格描述（标量或批量，空串=无描述）
    #   language           ：每条样本语言，None→"Auto"
    #   non_streaming_mode  ：默认 True（一次给全部文字）；False=模拟流式输入
    #   **kwargs            ：采样参数，详见 _merge_generate_kwargs
    # 返回：(wavs: List[np.ndarray], sample_rate: int)
    @torch.no_grad()
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        instruct: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the VoiceDesign model using natural-language style instructions.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            instruct:
                Instruction(s) describing desired voice/style. Empty string is allowed (treated as no instruction).
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`,
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)
        """
        # 仅 voice_design 模型支持。
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_design, Please check Model Card or Readme for more details."
            )

        # 统一成列表；language 缺省 "Auto"。
        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        instructs = self._ensure_list(instruct)

        # 单条 language/instruct 广播到 N 条样本。
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        # 批量长度必须一致。
        if not (len(texts) == len(languages) == len(instructs)):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}, instruct={len(instructs)}")

        self._validate_languages(languages)

        # 待合成文字 → input_ids（assistant 模板包裹）。
        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        # instruct 描述 → instruct_ids（user 段落），空描述则置 None。
        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        # 合并采样参数。
        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        # 调用底层 generate()，传 instruct_ids 控制音色风格。
        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        # 解码 codec → 波形。
        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    # custom voice model
    # ------------------------------------------------------------------
    # generate_custom_voice：用预置音色合成语音（仅 custom_voice 模型）
    # ------------------------------------------------------------------
    # 用途：从模型内置的 9 个预置音色（Vivian/Ryan/Ono_Anna 等）里选一个合成语音，
    #        可选附加 instruct 文本做风格微调。
    # 与 generate_voice_design 的区别：本方法传具体 speaker 名字（已注册的预置音色），
    #        而非自由描述。
    # 关键参数：
    #   text               ：待合成文字（标量或批量）
    #   speaker            ：预置音色名（大小写不敏感，会校验是否在支持列表）
    #   language           ：每条样本语言，None→"Auto"
    #   instruct           ：可选风格描述；0b6 小模型不支持 instruct（会自动置 None）
    #   non_streaming_mode  ：默认 True；False=模拟流式输入
    #   **kwargs            ：采样参数，详见 _merge_generate_kwargs
    # 返回：(wavs: List[np.ndarray], sample_rate: int)
    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the CustomVoice model using a predefined speaker id, optionally controlled by instruction text.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            speaker:
                Speaker name(s). Will be validated against `model.get_supported_speakers()` (case-insensitive).
            instruct:
                Optional instruction(s). If None, treated as empty (no instruction).
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`,
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If any speaker/language is unsupported or batch sizes mismatch.
        """
        # 仅 custom_voice 模型支持。
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_custom_voice, Please check Model Card or Readme for more details."
            )

        # 统一成列表；language 缺省 "Auto"。
        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        speakers = self._ensure_list(speaker)
        # 0b6 小模型不支持 instruct，强制置空（避免传入报错）。
        if self.model.tts_model_size in "0b6": # for 0b6 model, instruct is not supported
            instruct = None
        instructs = self._ensure_list(instruct) if isinstance(instruct, list) else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))

        # 单条参数广播到 N 条样本。
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(speakers) == 1 and len(texts) > 1:
            speakers = speakers * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        # 四个字段长度必须一致。
        if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, speaker={len(speakers)}, instruct={len(instructs)}"
            )

        # 校验语言和音色是否被模型支持（大小写不敏感）。
        self._validate_languages(languages)
        self._validate_speakers(speakers)

        # 待合成文字 → input_ids。
        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        # instruct → instruct_ids，空则置 None。
        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        # 合并采样参数。
        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        # 调用底层 generate()，传 speakers（预置音色名）+ instruct_ids 控制风格。
        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        # 解码 codec → 波形。
        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs


    # 查询当前模型支持的预置音色名列表（小写、排序）。模型不限制时返回 None。
    def get_supported_speakers(self) -> Optional[List[str]]:
        """
        List supported speaker names for the current model.

        This is a convenience wrapper around `model.get_supported_speakers()`.
        If the underlying model does not expose speaker constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported speaker names (lowercased), if available.
                - None if the model does not provide supported speakers.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return None
        return sorted(supported)


    # 查询当前模型支持的语言列表（小写、排序）。模型不限制时返回 None。
    def get_supported_languages(self) -> Optional[List[str]]:
        """
        List supported language names for the current model.

        This is a convenience wrapper around `model.get_supported_languages()`.
        If the underlying model does not expose language constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported language names (lowercased), if available.
                - None if the model does not provide supported languages.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return None
        return sorted(supported)
