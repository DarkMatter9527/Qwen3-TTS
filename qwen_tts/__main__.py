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
# 本文件是 `python -m qwen_tts` 的入口（即作为模块运行时的执行点）。
# qwen_tts 包本身不把推理/TTS 主流程放在这里，真正的可执行入口是
# `qwen-tts-demo` 命令（定义在 cli/demo.py）。运行 `python -m qwen_tts`
# 只是打印一段提示，告诉用户应该用哪个 CLI 命令。

def main():
    print(
        "qwen_tts package.\n"
        "Use CLI entrypoints:\n"
        "  - qwen-tts-demo\n"
    )

if __name__ == "__main__":
    main()
