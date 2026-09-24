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
# 本脚本演示 Qwen3-TTS "CustomVoice" 模型（预置音色）的用法。
# 模型：Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice，内置 9 个音色（如 Vivian / Ryan / Ono_Anna 等）。
# 与声音克隆 (VoiceClone) 模式不同：这里不需要参考音频，只需选择内置 speaker。
#
# 核心 API：tts.generate_custom_voice(text, language, speaker, instruct)
#   - text：待合成文本；language：语种；speaker：预置音色名；
#   - instruct：可选，用自然语言描述情感/语气（如"用特别愤怒的语气说"），传空串等同不约束。
#
# 运行：python examples/test_model_12hz_custom_voice.py
# 输出：当前目录生成 qwen3_tts_test_custom_single.wav 与 qwen3_tts_test_custom_batch_*.wav。
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/"

    # 模型加载：from_pretrained 是 HuggingFace 风格。
    #   - dtype=torch.bfloat16：bfloat16 半精度，省显存、对大模型足够。
    #   - attn_implementation="flash_attention_2"：FlashAttention-2，加速注意力计算。
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # -------- 单条合成：带 instruct 情感控制 --------
    # torch.cuda.synchronize() 用来对齐 GPU 计算完成后再计时，时间统计更准。
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_custom_voice(
        text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
        language="Chinese",
        speaker="Vivian",
        instruct="用特别愤怒的语气说",
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Single] time: {t1 - t0:.3f}s")

    sf.write("qwen3_tts_test_custom_single.wav", wavs[0], sr)

    # -------- 批量合成：部分 instruct 留空 --------
    # 文本/语种/说话人/指令都是 list，按位置一一对应。instruct 为空串表示不加情感约束。
    texts = ["其实我真的有发现，我是一个特别善于观察别人情绪的人。", "She said she would be here by noon."]
    languages = ["Chinese", "English"]
    speakers = ["Vivian", "Ryan"]
    instructs = ["", "Very happy."]

    torch.cuda.synchronize()
    t0 = time.time()

    # max_new_tokens=2048：限制生成最多 2048 个新 token，防止生成失控。
    wavs, sr = tts.generate_custom_voice(
        text=texts,
        language=languages,
        speaker=speakers,
        instruct=instructs,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Batch] time: {t1 - t0:.3f}s")

    for i, w in enumerate(wavs):
        sf.write(f"qwen3_tts_test_custom_batch_{i}.wav", w, sr)


if __name__ == "__main__":
    main()
