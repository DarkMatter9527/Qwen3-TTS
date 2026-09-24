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
# 本脚本演示 Qwen3-TTS "Base" 模型的声音克隆 (VoiceClone) 用法。
# 模型：Qwen/Qwen3-TTS-12Hz-1.7B-Base。
#
# 声音克隆有两种模式：
#   1) ICL 上下文学习模式 (x_vector_only_mode=False)：同时使用参考音频的
#      离散 token (ref_code) + 说话人向量 (x-vector) 作为 prompt，效果更好，
#      但必须提供参考音频对应的文本 ref_text。
#   2) x-vector only 模式 (x_vector_only_mode=True)：只用说话人向量，不需要 ref_text，
#      方便但克隆效果略差。
#
# 核心 API：
#   - tts.generate_voice_clone(text, language, ref_audio, ref_text, x_vector_only_mode=...)
#     直接传参考音频和文本，模型内部构建 prompt 后合成。
#   - tts.create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=...)
#     只构建可复用的 prompt（VoiceClonePromptItem 列表），不立即合成；
#     之后调用 generate_voice_clone(..., voice_clone_prompt=items) 复用，省去重复编码参考音频。
#
# 本脚本系统对比"单/批量参考 × 单/批量合成 × 直接调用 vs 先建 prompt"等组合，
# 并对 ICL 与 x-vector-only 两种模式各跑一遍，输出到 OUT_DIR。
#
# 运行：python examples/test_model_12hz_base.py
# 输出：qwen3_tts_test_voice_clone_output_wav/ 下若干 case*_*.wav。
import os
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def ensure_dir(d: str):
    """确保输出目录存在，不存在则创建。"""
    os.makedirs(d, exist_ok=True)


def run_case(tts: Qwen3TTSModel, out_dir: str, case_name: str, call_fn):
    """统一封装：计时 + 调用 call_fn 拿到 (wavs, sr) + 把每条 wav 写盘。
    call_fn 是一个零参 lambda/闭包，里面真正调用 generate_voice_clone。"""
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = call_fn()

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[{case_name}] time: {t1 - t0:.3f}s, n_wavs={len(wavs)}, sr={sr}")

    for i, w in enumerate(wavs):
        sf.write(os.path.join(out_dir, f"{case_name}_{i}.wav"), w, sr)


def main():
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base/"
    OUT_DIR = "qwen3_tts_test_voice_clone_output_wav"
    ensure_dir(OUT_DIR)

    # 模型加载（同其他示例）：bfloat16 + FlashAttention-2。
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # 参考音频：单条用一个 URL，批量用两个 URL。
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    ref_audio_path_2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"

    ref_audio_single = ref_audio_path_1
    ref_audio_batch = [ref_audio_path_1, ref_audio_path_2]

    # 参考音频对应的文本（按位置与 ref_audio 一一对应）。
    # ICL 模式下需要这些文本帮模型对齐音频内容；x-vector only 模式不需要但传了也无妨。
    ref_text_single = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    ref_text_batch = [
        "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        "甚至出现交易几乎停滞的情况。",
    ]

    # 想要合成的目标文本：单条 + 批量（中英文混合）。
    syn_text_single = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
    syn_lang_single = "Auto"

    syn_text_batch = [
        "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    ]
    syn_lang_batch = ["Chinese", "English"]

    # 采样参数（用于控制生成质量与多样性）：
    #   - do_sample=True：开启采样（非贪心）；
    #   - top_k=50 / top_p=1.0：从 top-k 候选 token 里再按累积概率 top-p 采样；
    #   - temperature=0.9：温度越高越随机；
    #   - repetition_penalty=1.05：抑制重复 token；
    #   - subtalker_*：模型内部"子说话人"网络的采样参数（用于多码本并行预测）。
    common_gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
    )

    # 对两种克隆模式各跑一遍：xvec_only=False 是 ICL 模式，True 是只用 x-vector。
    for xvec_only in [False, True]:
        mode_tag = "xvec_only" if xvec_only else "icl"

        # Case 1：单条参考 + 单条合成，直接调用 generate_voice_clone。
        run_case(
            tts, OUT_DIR, f"case1_promptSingle_synSingle_direct_{mode_tag}",
            lambda: tts.generate_voice_clone(
                text=syn_text_single,
                language=syn_lang_single,
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
                **common_gen_kwargs,
            ),
        )

        # Case 1b：先 create_voice_clone_prompt 构建 prompt，再用 voice_clone_prompt= 复用合成。
        # 与 Case 1 输出等价，但 prompt 可保存/复用，适合同一个音色多次合成。
        def _case1b():
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
            )
            return tts.generate_voice_clone(
                text=syn_text_single,
                language=syn_lang_single,
                voice_clone_prompt=prompt_items,
                **common_gen_kwargs,
            )

        run_case(
            tts, OUT_DIR, f"case1_promptSingle_synSingle_promptThenGen_{mode_tag}",
            _case1b,
        )

        # Case 2：单条参考 + 批量合成（一个音色同时合成多段文本）。
        run_case(
            tts, OUT_DIR, f"case2_promptSingle_synBatch_direct_{mode_tag}",
            lambda: tts.generate_voice_clone(
                text=syn_text_batch,
                language=syn_lang_batch,
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
                **common_gen_kwargs,
            ),
        )

        # Case 2b：先 prompt 再批量合成。
        def _case2b():
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
            )
            return tts.generate_voice_clone(
                text=syn_text_batch,
                language=syn_lang_batch,
                voice_clone_prompt=prompt_items,
                **common_gen_kwargs,
            )

        run_case(
            tts, OUT_DIR, f"case2_promptSingle_synBatch_promptThenGen_{mode_tag}",
            _case2b,
        )

        # Case 3：批量参考 + 批量合成（每段合成文本对应自己的参考音频，按位置匹配）。
        # 注意 x_vector_only_mode 此时也要传 list，与 ref_audio 一一对应。
        run_case(
            tts, OUT_DIR, f"case3_promptBatch_synBatch_direct_{mode_tag}",
            lambda: tts.generate_voice_clone(
                text=syn_text_batch,
                language=syn_lang_batch,
                ref_audio=ref_audio_batch,
                ref_text=ref_text_batch,
                x_vector_only_mode=[xvec_only, xvec_only],
                **common_gen_kwargs,
            ),
        )

        # Case 3b：批量参考也支持先构建 prompt 再合成。
        def _case3b():
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=ref_audio_batch,
                ref_text=ref_text_batch,
                x_vector_only_mode=[xvec_only, xvec_only],
            )
            return tts.generate_voice_clone(
                text=syn_text_batch,
                language=syn_lang_batch,
                voice_clone_prompt=prompt_items,
                **common_gen_kwargs,
            )

        run_case(
            tts, OUT_DIR, f"case3_promptBatch_synBatch_promptThenGen_{mode_tag}",
            _case3b,
        )


if __name__ == "__main__":
    main()
