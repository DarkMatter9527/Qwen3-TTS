# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin


# =============================================================================
# 文件总览：Qwen3-TTS 文本处理器（Processor）
# -----------------------------------------------------------------------------
# 用途：Qwen3TTSProcessor 是文本侧的预处理入口，包装 Qwen2TokenizerFast，
#       把用户输入的文字切成模型可用的 input_ids 等张量。它不直接处理音频，
#       音频侧（特征提取/说话人编码）由其他模块负责。
# 主要方法：
#   __call__            把文本经 tokenizer 切成 input_ids，返回 BatchFeature
#   batch_decode        批量把 token id 序列解码回文本（调用底层 tokenizer）
#   decode             把单条 token id 序列解码回文本
#   apply_chat_template 用 Jinja 模板把多轮对话格式化为模型输入
# 关键概念：
#   - tokenizer：把文字切成离散 token id 的工具；Qwen2TokenizerFast 是快速版
#   - BatchFeature：把多个张量打包成字典-like 容器，方便传给模型
#   - ProcessingKwargs：处理器调用参数的容器，统一管理 padding 等选项
# =============================================================================


class Qwen3TTSProcessorKwargs(ProcessingKwargs, total=False):
    # 处理器调用参数容器：定义 __call__ 时文本侧(text_kwargs)的默认参数。
    # total=False 表示所有字段都可缺省。
    _defaults = {
        "text_kwargs": {
            "padding": False,        # 是否把短句补齐到批次内最长长度（默认不补）
            "padding_side": "left",  # 补齐方向：左侧补 pad（自回归生成常用左补）
        }
    }

class Qwen3TTSProcessor(ProcessorMixin):
    # -----------------------------------------------------------------------------
    # Qwen3-TTS 文本处理器
    # -----------------------------------------------------------------------------
    # 继承自 HuggingFace 的 ProcessorMixin，内部持有一个 Qwen2TokenizerFast。
    # 职责：把文字切成 input_ids（__call__）、把 id 解码回文字（decode/batch_decode）、
    #       把多轮对话套用聊天模板（apply_chat_template）。是模型输入的"前置门卫"。
    # -----------------------------------------------------------------------------
    r"""
    Constructs a Qwen3TTS processor.

    Args:
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The text tokenizer.
            （中文：文本分词器，负责文字↔token id 的双向转换。
             这里用 Qwen2 系列的快速实现 Qwen2TokenizerFast）
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided, the default chat template is used.
            （中文：格式化多轮对话用的 Jinja 模板字符串。不传则用 tokenizer 自带的默认模板，
             把 system/user/assistant 等角色消息拼成模型能理解的一串文本）
    """

    attributes = ["tokenizer"]
    # tokenizer 类名候选：加载时会按此顺序尝试 Qwen2Tokenizer 与 Qwen2TokenizerFast
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self, tokenizer=None, chat_template=None
    ):
        super().__init__(tokenizer, chat_template=chat_template)

    def __call__(self, text=None, **kwargs) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. 

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
                （中文：待编码的文本。可以是单条字符串、字符串列表（一批），
                 或已经预切词的字符串列表（此时需设 is_split_into_words=True
                 以区分"一条预切词"和"一批普通字符串"两种含义））
        """
        # （中文：必须传入 text，否则报错）
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        # 合并调用参数：把用户传入的 kwargs 与 Qwen3TTSProcessorKwargs 的默认值合并，
        # 同时带上 tokenizer 初始化时的参数，保证编码行为一致
        output_kwargs = self._merge_kwargs(
            Qwen3TTSProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        # 单条文本也统一包成列表，走批量处理逻辑
        if not isinstance(text, list):
            text = [text]

        # 调用底层 Qwen2TokenizerFast 把文本切成 input_ids（含 attention_mask 等）
        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # 打包成 BatchFeature 容器返回，便于直接喂给模型；tensor_type 控制返回张量类型
        return BatchFeature(
            data={**texts_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        （中文：批量解码：把多条 token id 序列一次性还原成文本，直接转发给底层 tokenizer。）
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        （中文：单条解码：把一条 token id 序列还原成文本，直接转发给底层 tokenizer。）
        """
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        # 把多轮对话套用 Jinja 模板，格式化成模型输入。
        # 若传入的是单条对话（第一个元素是 dict），则包成一批，统一走批量处理。
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        # 暴露模型实际需要的输入名（如 input_ids、attention_mask），
        # 去重后返回，供训练/推理框架自动对齐字段。
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
            )
        )


__all__ = ["Qwen3TTSProcessor"]
