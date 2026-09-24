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
# 本脚本是 Qwen3-TTS SFT（监督微调）流程的"第三步：训练"。
# 它从 Base 模型出发，用带标签的"参考音频 + 目标音频码本"数据继续训练 talker，
# 最终把微调后的 speaker 写到 codec_embedding 的某个 speaker 槽位，
# 并把 config 的 tts_model_type 改成 "custom_voice"——也就是把声音克隆模型
# 微调成一个"预置音色"模型，推理时直接用 speaker 名调用，无需再传参考音频。
#
# 整体三步流程：
#   1) prepare_data.py：原始 jsonl -> 带 audio_codes 的训练 jsonl
#   2) dataset.py：定义 TTSDataset
#   3) 本脚本 sft_12hz.py：Accelerate + AdamW 训练 + 保存权重
#
# 训练循环要点（见下方逐行注释）：
#   - 用参考音频 mel 经 speaker_encoder 抽出说话人向量，detach 后填到第 6 位。
#   - talker forward 用 inputs_embeds 喂入，labels 是第 0 层 codec 的位移监督。
#   - 另外用 forward_sub_talker_finetune 训练 16 层码本中的其余 15 层（sub_talker）。
#   - 总 loss = talker loss + 0.3 * sub_talker loss。
#
# 运行示例：
#   python finetuning/sft_12hz.py --train_jsonl data/train_ready.jsonl \
#       --speaker_name my_speaker --output_model_path output
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

# 训练过程中保存"目标说话人向量"的全局变量：第一次前向时拿到参考音频的 speaker embedding，
# 训练结束时把它写进 codec_embedding.weight 的第 3000 行（speaker 槽位），
# 这样推理时通过 speaker_id=3000 就能用这个固定音色，无需再传参考音频。
target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()

    # Accelerate 是 HuggingFace 的多卡/混合精度训练包装器。
    #   - gradient_accumulation_steps=4：累积 4 步再更新一次（等效 batch_size=4*args.batch_size）；
    #   - mixed_precision="bf16"：训练用 bfloat16 混合精度省显存；
    #   - log_with="tensorboard"：把训练曲线写到 tensorboard。
    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    MODEL_PATH = args.init_model_path

    # 加载 Base 模型（用 bfloat16 + FlashAttention-2 加速）。
    # 注意：这里用的是 torch_dtype= 而不是 dtype=，与 examples 中略有差异但作用相同。
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # AutoConfig.from_pretrained：HF 风格读取 config.json，供 dataset 构造输入时拿特殊 token id。
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # 读训练数据 -> TTSDataset -> DataLoader（collate_fn 来自 dataset.py）。
    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # AdamW 优化器，weight_decay=0.01 防过拟合。
    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    # accelerator.prepare：把 model/optimizer/dataloader 包装成支持分布式+混合精度的版本。
    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            # accumulate：处理梯度累积，每 4 个 micro-step 才真正 backward+step。
            with accelerator.accumulate(model):

                # 从 batch 取出 dataset.collate_fn 产出的各张量。
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                # 用 speaker_encoder 把参考音频 mel -> 说话人向量 x-vector，detach 后不参与反传。
                # 第一次前向时把它存到全局 target_speaker_embedding，保存权重时要用。
                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                # 双通道 input_ids：通道0文本 token，通道1 codec token。
                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # 文本通道 embedding：文本 token -> 向量，乘 mask 屏蔽无效位置。
                # codec 通道 embedding：codec token -> 向量，再乘 mask；
                # 把第 6 位（speaker embedding 槽位）替换为说话人向量——这就是"克隆音色"被注入的位置。
                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                # 文本 + codec 两路 embedding 相加，得到 talker 的输入表示。
                input_embeddings = input_text_embedding + input_codec_embedding

                # 第 1~15 层 codec 码本各自有独立 embedding（code_predictor 里），逐层累加。
                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # talker 前向：输入用 [:-1] 位置（位移对齐），监督第 0 层 codec 的 [1:]。
                # labels 自动用 -100 屏蔽非监督位置，计算 CE loss。
                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                # 用 talker 最后一层 hidden state 喂给 sub_talker 训练其余 15 层码本。
                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                # 总 loss = talker 主 loss + 0.3 * sub_talker loss（权重可调）。
                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                # 梯度裁剪到 1.0 防爆。
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            # 每 10 步打印一次 loss（只在主进程 + 同步点）。
            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        # 每个 epoch 结束：在主进程上保存一份 checkpoint。
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            # 先把原始模型目录整体复制过来（含 config、tokenizer 等所有附属文件）。
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            # 改写 config.json：把模型类型改成 "custom_voice"，并注册新 speaker 的 id=3000。
            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # 解包模型，准备保存权重。
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            # speaker_encoder 不需要保存到推理权重里，删掉。
            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            # 关键步骤：把训练时记录的 target_speaker_embedding 写到 codec_embedding.weight 的第 3000 行，
            # 这样推理时调用 generate_custom_voice(speaker=args.speaker_name) 就能用这个固定音色，
            # 不需要再传参考音频——把"声音克隆"变成"预置音色"。
            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

if __name__ == "__main__":
    train()
