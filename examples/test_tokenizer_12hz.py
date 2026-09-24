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
# 本脚本演示如何单独使用 Qwen3TTSTokenizer（12Hz 帧率版本）进行音频编解码。
#
# 用途/位置：
#   Qwen3-TTS 由"音频编解码器 (tokenizer)"和"语言大模型 (talker)"两部分组成。
#   tokenizer 只负责把原始音频 ↔ 离散 token 互转，本身不做 TTS。
#   本脚本仅演示 tokenizer 的 encode / decode 各种输入输出形式，
#   方便理解 prepare_data.py、声音克隆流程里"参考音频是怎么变成 token"的。
#
# 运行：python examples/test_tokenizer_12hz.py
# 输出：在当前目录生成多个 decoded_*.wav 文件（与输入波形等价的还原结果）。
import io
import requests
import soundfile as sf

from qwen_tts import Qwen3TTSTokenizer

# 两段公开的演示音频（OSS URL），下面会作为输入反复使用。
audio_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_1.wav"
audio_2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_2.wav"

# -------- 单条输入：直接传 wav 路径/URL --------
# from_pretrained 是 HuggingFace 风格加载：从模型仓库名/本地路径加载权重和配置。
# device_map="cuda:0" 把 tokenizer 的模型放到 0 号 GPU。
tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
)

# encode：音频 -> token；decode：token -> 音频波形 + 采样率。
enc1 = tokenizer_12hz.encode(audio_1)
wavs1, out_sr1 = tokenizer_12hz.decode(enc1)
sf.write("decoded_single_12hz.wav", wavs1[0], out_sr1)

# -------- 批量输入：wav 路径列表 --------
# encode/decode 都支持一次处理多条音频，返回的 enc2.audio_codes 是多段 token 的列表。
enc2 = tokenizer_12hz.encode([audio_1, audio_2])
wavs2, out_sr2 = tokenizer_12hz.decode(enc2)
for i, w in enumerate(wavs2):
    sf.write(f"decoded_batch_12hz_{i}.wav", w, out_sr2)

# -------- 以 dict 作为 decode 输入 (12hz) --------
# 取第一条样本的 codes，用 {"audio_codes": tensor} 这种 dict 形式传给 decode。
# 这样可以只解码某一段 token，不必重新 encode。
dict_input_12hz = {"audio_codes": enc2.audio_codes[0]}  # torch.Tensor
wavs_d1, out_sr_d1 = tokenizer_12hz.decode(dict_input_12hz)
sf.write("decoded_dict_12hz.wav", wavs_d1[0], out_sr_d1)

# -------- 以 list[dict] 作为 decode 输入 (12hz) --------
# 把多段 token 包装成 list[dict] 再 decode，等价于批量解码。
list_dict_input_12hz = [{"audio_codes": c} for c in enc2.audio_codes]  # list of torch.Tensor
wavs_d2, out_sr_d2 = tokenizer_12hz.decode(list_dict_input_12hz)
for i, w in enumerate(wavs_d2):
    sf.write(f"decoded_listdict_12hz_{i}.wav", w, out_sr_d2)

# -------- 以 numpy 形式作为 decode 输入 (12hz) --------
# 把 token 转成 numpy 模拟"已经序列化保存过的载荷"，再解码——演示跨类型兼容性。
list_dict_numpy_12hz = [{"audio_codes": c.cpu().numpy()} for c in enc2.audio_codes]
wavs_d3, out_sr_d3 = tokenizer_12hz.decode(list_dict_numpy_12hz)
for i, w in enumerate(wavs_d3):
    sf.write(f"decoded_listdict_numpy_12hz_{i}.wav", w, out_sr_d3)

# -------- 直接传 numpy 波形（必须额外传采样率 sr） --------
# 当输入不是路径而是已经读好的波形数组时，encode 无法从数组推断采样率，
# 所以必须显式 sr= 参数。这里先用 requests+soundfile 把 wav 读成 (y, sr)。
data = requests.get(audio_2, timeout=30).content
y, sr = sf.read(io.BytesIO(data))
enc3 = tokenizer_12hz.encode(y, sr=sr)
wavs3, out_sr3 = tokenizer_12hz.decode(enc3)
sf.write("decoded_numpy_12hz.wav", wavs3[0], out_sr3)