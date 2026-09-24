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
qwen_tts: Qwen-TTS 包入口文件。

本文件是 qwen_tts 包的对外接口，只负责"重新导出"内部模块里的关键类，
让用户可以直接 `from qwen_tts import Qwen3TTSModel` 这种简短写法使用。

包结构概览：
  - inference/qwen3_tts_model.py：Qwen3TTSModel 主类（端到端 TTS 推理入口），
    提供 generate_custom_voice / generate_voice_design / generate_voice_clone
    以及 create_voice_clone_prompt 等方法。VoiceClonePromptItem 是声音克隆
    时复用的 prompt 数据结构。
  - inference/qwen3_tts_tokenizer.py：Qwen3TTSTokenizer（音频编解码器），
    只负责把音频编/解码为离散 token，不处理文本。

典型用法（详见各 examples 脚本）：
  tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base",
          device_map="cuda:0", dtype=torch.bfloat16,
          attn_implementation="flash_attention_2")
"""

from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

__all__ = ["__version__"]