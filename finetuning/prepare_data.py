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
# 本脚本是 Qwen3-TTS SFT（监督微调）流程的"第一步：数据准备"。
# 它把原始 jsonl（每行含 audio 路径、text、ref_audio 等字段）转成训练用 jsonl，
# 关键是用 Qwen3TTSTokenizer 把每条目标音频预先编码成离散 token (audio_codes)，
# 写回每行，供 dataset.py 在训练时直接读用，避免训练时反复跑 tokenizer。
#
# 微调整体三步流程：
#   1) 本脚本 prepare_data.py：原始 jsonl -> 带 audio_codes 的训练 jsonl
#   2) dataset.py：定义 TTSDataset，把 jsonl 一条变成模型可吃的 batch 张量
#   3) sft_12hz.py：用 Accelerate + AdamW 训练，最后保存微调后的权重
#
# 输入 jsonl 每行示例：
#   {"audio": "<wav 路径或 URL>", "text": "...", "ref_audio": "...", "language": "Chinese"}
# 输出 jsonl 每行示例（多了 audio_codes 字段）：
#   { ..., "audio_codes": [[token...], [token...], ... 共16层码本] }
#
# 运行示例：
#   python finetuning/prepare_data.py \
#       --input_jsonl data/train_raw.jsonl \
#       --output_jsonl data/train_ready.jsonl

import argparse
import json

from qwen_tts import Qwen3TTSTokenizer

# 批量推理大小：每凑够这么多条音频就调一次 tokenizer.encode，避免一条条调用效率低。
BATCH_INFER_NUM = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    # 加载音频编解码器（只负责音频 ↔ token，与文本无关）。
    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    # 一次性读入所有训练行并解析为 dict 列表。
    total_lines = open(args.input_jsonl).readlines()
    total_lines = [json.loads(line.strip()) for line in total_lines]

    final_lines = []
    batch_lines = []
    batch_audios = []
    # 累积 BATCH_INFER_NUM 条音频后批量 encode，再把 audio_codes 写回对应行。
    for line in total_lines:

        batch_lines.append(line)
        batch_audios.append(line['audio'])

        if len(batch_lines) >= BATCH_INFER_NUM:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, line in zip(enc_res.audio_codes, batch_lines):
                line['audio_codes'] = code.cpu().tolist()
                final_lines.append(line)
            batch_lines.clear()
            batch_audios.clear()

    # 处理最后不足 BATCH_INFER_NUM 的尾巴。
    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, line in zip(enc_res.audio_codes, batch_lines):
            line['audio_codes'] = code.cpu().tolist()
            final_lines.append(line)
        batch_lines.clear()
        batch_audios.clear()

    # 重新序列化为 jsonl 写盘（ensure_ascii=False 保留中文可读）。
    final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]

    with open(args.output_jsonl, 'w') as f:
        for line in final_lines:
            f.writelines(line + '\n')

if __name__ == "__main__":
    main()
