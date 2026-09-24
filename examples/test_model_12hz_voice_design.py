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
# 本脚本演示 Qwen3-TTS "VoiceDesign" 模型（自然语言描述生成音色）的用法。
# 模型：Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign。
# 与 CustomVoice 不同：这里没有预置音色名，而是用一段自然语言 instruct
# 描述想要的音色特征（音色、音调、情绪、年龄感等），模型据此合成。
#
# 核心 API：tts.generate_voice_design(text, language, instruct)
#
# 运行：python examples/test_model_12hz_voice_design.py
# 输出：当前目录生成 qwen3_tts_test_voice_design_single.wav 与 batch_*.wav。
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/"

    # 模型加载（同 custom_voice 示例）：bfloat16 + FlashAttention-2。
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # -------- 单条合成：用 instruct 描述音色 --------
    # 例：要求生成"撒娇稚嫩的萝莉女声，音调偏高且起伏明显"。
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_voice_design(
        text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
        language="Chinese",
        instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[VoiceDesign Single] time: {t1 - t0:.3f}s")

    sf.write("qwen3_tts_test_voice_design_single.wav", wavs[0], sr)

    # -------- 批量合成：中英文 + 不同音色描述 --------
    # 文本/语种/instruct 都是 list 按位置对应。
    texts = [
        "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
        "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
    ]
    languages = ["Chinese", "English"]
    instructs = [
        "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
        "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
    ]

    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_voice_design(
        text=texts,
        language=languages,
        instruct=instructs,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[VoiceDesign Batch] time: {t1 - t0:.3f}s")

    for i, w in enumerate(wavs):
        sf.write(f"qwen3_tts_test_voice_design_batch_{i}.wav", w, sr)


if __name__ == "__main__":
    main()