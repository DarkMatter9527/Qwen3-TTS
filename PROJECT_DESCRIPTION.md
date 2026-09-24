# Qwen3-TTS 学习指导

> 本文档面向初学者，尽量用通俗语言解释专业术语。读完后你应能说清楚：模型长什么样、为什么这么设计、一段文字是怎么变成一段语音的、以及怎么用这套代码。

---

## 一、先建立直觉：TTS（文字转语音）到底要解决什么问题

TTS = Text To Speech，把文字变成语音。难点不在于"读出来"，而在于读得**自然**：
- 同一句"你好"，老人、小孩、男声、女声读出来不一样；
- 同一个人，开心、生气、平静地读，韵律、语调也不同；
- 长句子要换气、有停顿、有重音。

所以一个现代 TTS 系统通常要回答三个问题：
1. **说什么内容**（由输入文字决定）；
2. **用谁的声音说**（音色 / 说话人身份）；
3. **用什么语气说**（情感、语速、韵律）。

Qwen3-TTS 用一个端到端的大模型把这三件事一次搞定，而不像传统方案要拼好几个模块。

---

## 二、Qwen3-TTS 的整体结构（自顶向下）

整个仓库可以拆成 3 大块，对应 3 个目录：

```
qwen_tts/
├── inference/          # 第 1 块：对外暴露的"好用的接口"（用户直接调这里）
│   ├── qwen3_tts_model.py      # Qwen3TTSModel：高层 API（generate_custom_voice 等）
│   └── qwen3_tts_tokenizer.py # Qwen3TTSTokenizer：语音 ↔ 离散编码 的高层封装
├── core/
│   ├── models/          # 第 2 块：核心大模型（Talker 语言模型 + 说话人编码器）
│   │   ├── modeling_qwen3_tts.py        # 主模型 Qwen3TTSForConditionalGeneration
│   │   ├── configuration_qwen3_tts.py   # 模型配置（超参数）
│   │   └── processing_qwen3_tts.py      # 文本 tokenizer 处理器
│   └── tokenizer_25hz/ # 第 3 块：语音 tokenizer（把音频压成编号、再还原）
       ├── tokenizer_12hz/  # 正式发布版（12.5 帧/秒，多码本 16×2048）
       └── tokenizer_25hz/  # 早期版（25 帧/秒，单码本 32768，DiT+BigVGAN）
```

一句话概括这三块的关系：

> **第 3 块（语音 tokenizer）** 负责"音频 ↔ 编号"的翻译；
> **第 2 块（Talker 大模型）** 负责"文字 + 条件 → 编号序列"的生成；
> **第 1 块（inference 封装）** 把上面两块串起来，给用户一个简单好用的函数。

---

## 三、模型结构原理详解

### 3.1 核心思想：把语音变成"文字"，然后让 LLM 来"说话"

Qwen3-TTS 最关键的思路是：**语音也能像文字一样被"分词"成一串编号（token）**。

- 文字 LLM（比如 Qwen）是把"今天天气不错"切成 `[今, 天, 天, 气, 不, 错]` 这些字，然后预测下一个字。
- Qwen3-TTS 把一段语音也切成一串编号（比如 `[37, 1024, 8, ...]`），然后用同样的"预测下一个 token"的方式去生成语音。

这样做的好处是：可以直接复用成熟的 LLM 架构和训练方法，语音和文字用同一套"语言"来建模，所以叫"端到端"。

- **提问：离散编号是就像tokenID，需要用离散编号去找对应的“语音向量“？？？**

  **离散编号就像 token ID，本身只是“词表/码本里的行号”；要用它建模或合成，必须查表变成连续向量。**

  - 第一步预测语音token

    语音 token ID 类似文本token ID：

    ```
    语音 token ID -> speech embedding -> 语音向量 -> Transformer -> 预测下一个 token 的 logits，得到语音 token ID
    ```

    这个“语音向量”通常叫：

    - codebook vector：码本向量
    - audio embedding：音频嵌入
    - latent / 潜变量：压缩后的声学表示

    它不是最终波形，而是某种中间表示，可能编码了音高、音色、内容、韵律等信息。

  - 第二步：把预测出的 token 还原成语音

    预测出一串语音 token ID 后：

    ```
    [37, 1024, 8, ...] 
    -> 查声学码本，得到一串语音向量 
    -> decoder / vocoder 
    -> 波形
    ```

    这里查的是 **音频 codec 的码本**。

- **提问：speech embedding 和 codec 码本共享分别是什么？？都是向量化的音频信息？？如果都是向量化的音频信息，共享不就好了吗，为什么还要在输入和输出的时候做区分？？？**

  - **codec是什么？？**

    音频 codec 先把波形压缩成连续 latent(latent 就是神经网络学出来的压缩版特征向量,文本里的 embedding（词嵌入向量）本质也是一种 latent)，然后做向量量化：找码本里最接近的向量，记下它的编号。

    ```
    音频片段 -> encoder -> 连续 latent -> 找最近码本向量 -> token ID
    ```

    解码时反过来：

    ```
    token ID -> 查 codec 码本 -> 声学向量 -> decoder/vocoder -> 波形
    ```

    所以 **codec 码本**里的每一行是一个“量化后的声学 latent 向量”，它直接服务于音频重建。

### 3.2 语音 tokenizer：把声音压成编号，再还原成声音

这部分对应 `core/tokenizer_*/`。它要解决两个方向：
- **encode（编码）**：音频波形 → 一串离散编号（codes）
- **decode（解码）**：一串编号 → 音频波形

仓库里有两个版本：

#### （1）V2 = 12Hz 版（正式发布版，推荐使用）

文件：`core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`

- 输入采样率 24kHz（每秒 24000 个采样点）；
- 每秒语音被压成 **12.5 个时间步**（所以叫 12Hz）；
- 每个时间步用 **16 个编号**共同表示，每个编号从一个 **2048 项的字典（codebook）**里选；
- 所以每秒语音 ≈ 12.5 × 16 = 200 个编号。

它内部结构：
- **Encoder（编码器）**：基于 **Mimi**（一种语音 tokenizer 架构）。先用卷积把波形压成特征，再用 Transformer 提取语义信息，最后用 **RVQ（残差向量量化）** 把连续特征切成 16 层编号。
- **Decoder（解码器）**：一个 Transformer + 卷积声码器，把 16 层编号还原成波形。用了 **SnakeBeta** 激活函数（对音频信号更友好的激活函数）。

> 通俗解释 **RVQ（残差向量量化）**：就像画画。第一本字典负责画大致轮廓（低分辨率），第二本字典负责修正轮廓的误差，第三本修正更细的误差……叠 16 层就能还原得很精细。每一层叫一个 codebook，每一层选出的编号叫一个 quantizer 的 code。

> 通俗解释 **codebook（码本）**：可以理解成一本"字典"，里面存了 2048 个"标准声音碎片"。编码就是找出每个时刻每层字典里最像的那个碎片的编号；解码就是按编号把碎片取出来拼回去。

#### （2）V1 = 25Hz 版（早期版本，未对外发布）

文件：`core/tokenizer_25hz/modeling_qwen3_tts_tokenizer_v1.py`

- 每秒 25 个时间步，每个时间步只有 **1 个编号**，但字典很大（32768 项）；
- 编码器用的是 **Whisper 风格的 Transformer**（`vq/whisper_encoder.py`），量化用 **GRVQ**（`vq/speech_vq.py`、`vq/core_vq.py`）；
- 解码器是 **DiT（Diffusion Transformer，扩散模型）+ BigVGAN（声码器）**：先用扩散模型从编号生成 mel 频谱，再用 BigVGAN 把 mel 频谱变成波形。

> 通俗解释 **mel 频谱图**：把声音按"时间 × 频率"画成一张二维热图，频率轴按人耳感知（mel 刻度）拉伸，让人耳敏感的频率更清楚。

> 通俗解释 **DiT（扩散模型）**：一种生成模型，先往数据里加噪声，再学习一步步去噪还原图像/频谱。这里用它从编号生成 mel。

> 通俗解释 **BigVGAN**：一个"声码器（vocoder）"，专门把 mel 频谱图变回能听的波形。

由于 12Hz 版用更轻的非扩散解码器就达到了更好的音质，所以正式发布的是 12Hz 版。

### 3.3 Talker 大模型：从文字生成语音编号

这部分对应 `core/models/modeling_qwen3_tts.py`。核心类是 `Qwen3TTSForConditionalGeneration`，它由三部分组成：

```
Qwen3TTSForConditionalGeneration
├── talker        # 主 LLM，Qwen3 风格的 Transformer，负责生成"第一层编号"
│   ├── model                 # Transformer 主体（含 codec_embedding、text_embedding）
│   ├── text_projection       # 把文字 embedding 投影到 talker 的维度
│   ├── codec_head            # 线性层，预测每个时间步的第 0 层编号
│   └── code_predictor        # ⭐ 子模型（sub-talker），预测同一时间步的第 1~15 层编号
├── speaker_encoder # 仅 Base 模型有：ECAPA-TDNN，从参考音频抽说话人特征
└── speech_tokenizer # 上面 §3.2 的语音 tokenizer，运行时注入
```

> 通俗解释 **embedding（嵌入向量）**：把一个编号（比如"编号 37"）变成一串连续的小数（一个向量），让神经网络能处理。可以理解成"编号的电话号码"被换成"编号的性格描述"。

> 通俗解释 **Transformer / 注意力（Attention）**：神经网络里的一种结构，让每个位置的信息都能"看到"其它位置的信息并加权融合。LLM 靠它理解上下文。

#### 3.3.1 Talker 主模型（生成第 0 层编号）

- 架构：Qwen3 风格的 decoder-only Transformer（就是和 Qwen 文字大模型同款）。
- 它的输入是一个"超长 embedding 序列"，里面把以下信息拼在一起：
  - 控制标签（是否思考 / 语言 / 说话人身份）；
  - 角色 token（`<|im_start|>assistant`）；
  - 文字内容 embedding（要合成的话）；
  - 参考音频的编号 embedding（仅 ICL 模式，给模型"示范"）；
  - 起始标记。
- 它**自回归地**（一步一步）预测每个时间步的 **第 0 层编号**（用 `codec_head`）。

> 通俗解释 **自回归（autoregressive）**：像写文章一样，写完一个字再看下一个字该写啥。模型每生成一个 token，就把它喂回去生成下一个。

#### 3.3.2 Code Predictor / Sub-talker（生成第 1~15 层编号）⭐

这是 Qwen3-TTS 最关键的设计之一，叫"**离散多码本 LM**"。

- 它是一个**很小的 Transformer**（看配置 `Qwen3TTSTalkerCodePredictorConfig`：5 层、hidden 1024，比主 talker 小得多）。
- 每当主 talker 生成了一个时间步的第 0 层编号后，code_predictor 拿这个编号 + 主 talker 的隐藏状态，**一次性预测剩下 15 层编号**。
- 所以每个时间步的 16 层编号是这样来的：
  - 第 0 层：主 talker 自回归生成（最重要，决定内容/韵律主框架）；
  - 第 1~15 层：sub-talker 在第 0 层基础上一次性补齐（负责细化音色/细节）。

> 通俗解释 **多码本（multi-codebook）**：每个时间步的语音用 16 个编号共同描述。第 0 层像"骨架"，后 15 层像"血肉"。骨架由大模型精雕细琢（慢但准），血肉由小模型批量补（快），兼顾质量和速度。

> 通俗解释 **多 token 预测（Multi-Token Prediction, MTP）**：一次预测多个 token，而不是一个一个慢慢来。这里 sub-talker 就是 MTP 的实现。

#### 3.3.3 说话人编码器（Speaker Encoder，仅 Base 模型）

- 类：`Qwen3TTSSpeakerEncoder`，基于 **ECAPA-TDNN** 论文。
- 作用：把一段参考音频压缩成一个**固定长度的向量**（叫 speaker embedding / x-vector），代表"这是谁的声音"。
- 用于"声音克隆"：给 3 秒参考音频，提取说话人向量，模型就能用这个人的声音说新内容。

> 通俗解释 **x-vector / speaker embedding**：把一段声音里的"是谁在说话"这个信息，浓缩成一个固定长度的"身份证号向量"。模型拿这个向量去模仿音色。

> 通俗解释 **ECAPA-TDNN**：一种专门提取说话人特征的网络结构。用了 TDNN（时延神经网络，按时间卷积）、Res2Net（多尺度残差块）、SE（通道注意力，让网络自己决定哪些通道重要）、Attentive Statistics Pooling（带注意力的统计池化，把变长序列压成定长向量）。

### 3.4 三种模型类型（tts_model_type）

配置里的 `tts_model_type` 决定模型能做什么：

| 类型 | 类别 | 用法 | 说话人来源 | 语气控制 |
|---|---|---|---|---|
| `base` | 声音克隆 | `generate_voice_clone` | 参考音频（speaker_encoder + ICL） | 否 |
| `custom_voice` | 预置音色 | `generate_custom_voice` | 9 个内置音色之一（speaker id） | 1.7B 支持 instruct |
| `voice_design` | 声音设计 | `generate_voice_design` | 自然语言描述生成 | 是（instruct） |

- **CustomVoice**：内置 9 个音色（Vivian、Ryan、Ono_Anna 等），每个音色对应一个 id，模型用 id 的 embedding 当"说话人身份证"。
- **VoiceDesign**：你用自然语言描述"一个 17 岁少年、紧张、声音发抖"，模型直接合成符合描述的声音。
- **Base**：给 3 秒参考音频就能克隆音色，支持两种模式：
  - **ICL 模式**（In-Context Learning，上下文学习）：把"参考音频的编号 + 参考文字"作为示例拼到输入里，模型模仿着说新内容。质量高，但需要 ref_text。
  - **x-vector only 模式**：只用说话人向量，不需要 ref_text。更省事，但克隆质量略低。

> 通俗解释 **ICL（上下文学习）**：就像给模型看一个"例题"——"这段音频念的是'你好'，念成这样；现在请你用同样的嗓音念'再见'"。模型从例题里"举一反三"。

---

## 四、从用户输入到输出的完整运转流程

下面以 **Base 模型 + ICL 声音克隆** 为例，走一遍从输入到音频输出的全过程。其它模式只是省略其中几步。

### 第 0 步：用户调用

```python
model.generate_voice_clone(
    text="今天天气不错",
    language="Chinese",
    ref_audio="clone.wav",
    ref_text="Okay. Yeah. I resent you."
)
```

入口在 `inference/qwen3_tts_model.py` 的 `generate_voice_clone`。

### 第 1 步：构建声音克隆 prompt（create_voice_clone_prompt）

位置：`inference/qwen3_tts_model.py`

1. 读取参考音频（支持本地路径 / URL / base64 / numpy 数组），统一成 `(waveform, sr)`；
2. 用 `speech_tokenizer.encode(...)` 把参考音频压成 `ref_code`（12Hz 版形状是 `(T, 16)`，T 是时间步数）；
3. 把参考音频重采样到 24kHz，用 `speaker_encoder` 提取 `ref_spk_embedding`（一个定长向量）；
4. 打包成 `VoiceClonePromptItem(ref_code, ref_spk_embedding, x_vector_only_mode=False, icl_mode=True, ref_text)`。

### 第 2 步：构建文字输入

把要合成的话包成对话格式：
```
<|im_start|>assistant
今天天气不错<|im_end|>
<|im_start|>assistant
```
然后用文本 tokenizer（Qwen2TokenizerFast）切成 `input_ids`。

参考文字也包成 `<|im_start|>assistant\n{ref_text}<|im_end|>\n` 并切分成 `ref_ids`。

### 第 3 步：调用底层 `model.generate(...)`

位置：`core/models/modeling_qwen3_tts.py` 的 `Qwen3TTSForConditionalGeneration.generate`。

这里会逐样本拼接 **talker 输入 embedding 序列**，顺序大致是：

```
[控制标签(think/nothink + language + speaker_emb + pad + bos)]   ← codec prefill
+ [<|im_start|>assistant]                                          ← role tokens
+ [参考文字 emb + 参考音频编号 emb（ICL 对齐） + 目标文字 emb]      ← ICL prompt
```

具体细节：
- **控制标签**：决定模型"要不要思考"（think/nothink）、说什么语言（language_id）、用谁的声音（speaker embedding 或克隆向量）。
- **ICL 拼接**：`generate_icl_prompt` 把参考文字 embedding 和参考音频编号 embedding 在时间上对齐相加，让模型知道"这段文字对应这段声音"；目标文字再跟在后面。
  - `non_streaming_mode=True`：参考文字和编号同时喂入（非流式）；
  - `non_streaming_mode=False`：模拟流式，文字和编号交错喂入（"双轨混合流式"）。
- **左 padding**：把 batch 内不同长度的样本 pad 成等长，方便并行推理。

### 第 4 步：Talker 自回归生成

调用 `self.talker.generate(...)`（HF `GenerationMixin`），进入标准的"预填充 + 逐步解码"循环：

1. **Prefill 阶段**：一次性把第 3 步拼好的长 embedding 序列喂进去，算出 KV 缓存（`past_key_values`）。
2. **逐步解码**：每生成一个 token：
   - 主 talker Transformer 前向，用 `codec_head` 预测 **第 0 层编号** 的 logits，采样（top-k / top-p / temperature）得到第 0 层编号；
   - 把第 0 层编号 + 主 talker 的隐藏状态喂给 **code_predictor（sub-talker）**，它再预测 **第 1~15 层编号**；
   - 把这 16 层编号拼起来当作当前时间步的输出；
   - 把输出反馈回去，继续预测下一个时间步的第 0 层，直到遇到 `codec_eos_token`（结束标记）。

> 通俗解释 **KV 缓存（past_key_values）**：Transformer 每次算注意力都要用到之前所有 token 的 key/value。为了避免每步都重算，把这些存起来复用，这就是 KV cache。

> 通俗解释 **top-k / top-p / temperature**：采样参数。temperature 越高越随机；top-k 只在概率最高的 k 个里选；top-p 只在累积概率 ≤ p 的那些里选。它们一起控制生成结果的"多样性 vs 稳定性"。

### 第 5 步：后处理生成编号

`generate` 返回 `talker_codes_list`，每个样本是一个 `(T, 16)` 的编号张量（T 是生成的时间步数，16 是 codebook 层数）。

- 在第一层编号里找 `codec_eos_token`，按它截断；
- 如果是 ICL 模式，把 `ref_code` 拼到生成编号前面（因为解码时要还原完整音频再切掉参考段）。

### 第 6 步：解码成音频

调用 `speech_tokenizer.decode([{audio_codes: c} for c in codes])`：
- 12Hz 版：把 `(T, 16)` 编号喂给 Mimi 解码器，直接还原成波形；
- 输出 `wavs`（float32 numpy 数组列表）和采样率 `sr`。

最后在 `generate_voice_clone` 里按 ref_code 长度切掉参考段，返回最终的 `wavs` 和 `sr`。

用户拿到波形后用 `soundfile.write("out.wav", wavs[0], sr)` 保存即可。

---

## 五、如何使用这套代码

### 5.1 安装

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
# 可选：装 FlashAttention 2 省显存
pip install -U flash-attn --no-build-isolation
```

本地开发模式：
```bash
cd Qwen3-TTS
pip install -e .
```

### 5.2 三种典型用法（直接抄）

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

# 公共加载方式（模型名换成对应的即可）
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",   # 或 VoiceDesign / Base
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

#### 用法 A：预置音色（CustomVoice）
```python
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese",     # 传 "Auto" 自动识别
    speaker="Vivian",
    instruct="用特别愤怒的语气说",   # 1.7B 才支持，0.6B 传 None
)
sf.write("out.wav", wavs[0], sr)
```

#### 用法 B：声音设计（VoiceDesign）
```python
wavs, sr = model.generate_voice_design(
    text="哥哥，你回来啦！",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。",
)
sf.write("out.wav", wavs[0], sr)
```

#### 用法 C：声音克隆（Base）
```python
wavs, sr = model.generate_voice_clone(
    text="I am solving the equation.",
    language="English",
    ref_audio="clone.wav",          # 路径 / URL / base64 / (np_array, sr)
    ref_text="Okay. Yeah. I resent you.",   # x_vector_only_mode=True 时可省略
)
sf.write("out.wav", wavs[0], sr)
```

复用同一段参考（避免每次重算特征）：
```python
prompt = model.create_voice_clone_prompt(ref_audio=..., ref_text=...)
wavs, sr = model.generate_voice_clone(text=[...], language=[...], voice_clone_prompt=prompt)
```

#### 用法 D：只编解码音频（Tokenizer）
```python
from qwen_tts import Qwen3TTSTokenizer
tok = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
enc = tok.encode("https://.../demo.wav")
wavs, sr = tok.decode(enc)
```

### 5.3 本地 Web UI

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000
```
然后浏览器打开 `http://localhost:8000`。Base 模型远程访问建议加 HTTPS（见 README）。

### 5.4 关键参数说明

| 参数 | 作用 | 推荐值 |
|---|---|---|
| `dtype` | 模型精度，bf16 省显存 | `torch.bfloat16` |
| `attn_implementation` | 注意力实现，flash_attention_2 最快 | `flash_attention_2` |
| `max_new_tokens` | 最多生成多少个时间步 | 2048（评测默认） |
| `temperature` | 越高越随机 | 0.9 |
| `top_k` / `top_p` | 采样截断 | 50 / 1.0 |
| `repetition_penalty` | 抑制重复 | 1.05 |
| `non_streaming_mode` | True=非流式（一次性给文字）；False=模拟流式输入 | 非流式场景 True |

### 5.5 微调

见 `finetuning/` 目录：`prepare_data.py`（数据准备）→ `dataset.py`（数据集）→ `sft_12hz.py`（SFT 训练脚本）。细节看 `finetuning/README.md`。

### 5.6 vLLM 部署

vLLM-Omni 已 day-0 支持，详见 README 的 vLLM Usage 章节。

---

## 六、术语速查表（通俗版）

| 术语 | 通俗解释 |
|---|---|
| 采样率 (sample rate) | 每秒采集多少个声音点，24kHz = 每秒 24000 点 |
| mel 频谱图 | 声音的"时间×频率"二维表示，频率轴按人耳感知拉伸 |
| codebook（码本） | 一本存了 2048 个"标准声音碎片"的字典 |
| VQ（向量量化） | 把连续特征近似成字典里最像的某个编号 |
| RVQ（残差向量量化） | 多本字典层层逼近，第一本记大概，后面记越来越细的误差 |
| 多码本 LM | 每个时间步用 16 个编号共同描述，大模型预测骨架、小模型补细节 |
| embedding（嵌入） | 把编号变成神经网络能处理的连续向量 |
| Transformer / Attention | 让每个位置都能"看到"其它位置并加权融合的网络结构 |
| 自回归 | 写完一个 token 再写下一个，像写文章 |
| KV 缓存 | 存下之前算过的 key/value，避免每步重算 |
| GQA（分组查询注意力） | 多个查询头共享一组 key/value，省显存 |
| RoPE（旋转位置编码） | 用旋转把"位置"编进向量，让模型知道顺序 |
| 滑动窗口注意力 | 只看局部窗口内的 token，省算力 |
| x-vector / speaker embedding | 把"是谁的声音"浓缩成的定长向量 |
| ECAPA-TDNN | 提取说话人特征的网络（TDNN+Res2Net+SE+AttentiveStatsPooling） |
| ICL（上下文学习） | 把"例题"喂给模型让它模仿 |
| DiT | 扩散 Transformer，从噪声逐步去噪生成 mel（25Hz 版用） |
| BigVGAN | 声码器，把 mel 频谱变回波形（25Hz 版用） |
| Mimi | 一种语音 tokenizer 架构（卷积+Transformer+RVQ，12Hz 版用） |
| SnakeBeta | 一种对音频信号更友好的激活函数 |
| top-k / top-p / temperature | 采样参数，控制生成的随机性 |
| DiT / 非流式 vs 流式 | 非流式一次给全部文字；流式模拟"边说边输入文字"，延迟更低 |

---

## 七、推荐学习路径

1. 先读本文档建立全局观；
2. 看 `examples/test_model_12hz_*.py` 三个例子，跑通任一用法；
3. 读 `inference/qwen3_tts_model.py`（高层 API，最容易懂）；
4. 读 `core/models/configuration_qwen3_tts.py`（看模型有哪些旋钮）；
5. 读 `core/models/modeling_qwen3_tts.py` 的 `Qwen3TTSForConditionalGeneration.generate`（核心流程）；
6. 读 `core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`（语音 tokenizer）；
7. 进阶：读 `core/tokenizer_25hz/vq/` 三个文件（VQ 细节）和 `finetuning/`（怎么微调）。

每个 .py 文件都已添加详细中文注释，配合本文档阅读即可。
