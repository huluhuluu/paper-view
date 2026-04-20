---
title: "LoRA 高效微调技术深度解析"
date: 2026-04-02T16:30:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "lora-finetuning"
tags: ["LoRA", "PEFT", "fine-tuning", "LLM"]
categories: ["paper"]
math: true
---

# LoRA 高效微调技术深度解析

这几篇工作都在讨论同一个核心问题：低秩适配确实把可训练参数压下来了，但收敛方向、底座显存和初始化方式如果处理不好，LoRA 路线并不会自然得到最优的训练效率。

- **LoRA: Low-Rank Adaptation of Large Language Models** 正文引言先把问题落在部署和多任务切换成本上：当 GPT-3 这类模型达到 `175B` 规模后，为每个下游任务保存一份完整微调权重已经从“不方便”变成了真正的系统负担。论文的核心分析建立在低 intrinsic rank 假设上，即适配过程中真正需要学习的 `∆W` 并不一定需要满秩表达，因此正文把重点从“直接训练整块权重”转向“只训练一个低秩更新子空间”。方法部分把原始线性层改写为 `W0 + BA`，冻结 `W0`，只训练低秩矩阵 `A` 和 `B`，并主要作用在注意力投影层；同时正文强调，这个增量在部署时可以合并回原权重，因此不会引入额外推理延迟。实验部分显示，LoRA 在 RoBERTa、DeBERTa、GPT-2 与 GPT-3 上都能以远小于全量微调的参数量达到相当甚至更优的结果，在 GPT-3 175B 场景下可把可训练参数压到原模型的约 `0.01%`。

- **QLoRA: Efficient Finetuning of Quantized LLMs** 正文引言进一步把问题推进到显存极限：即便采用 LoRA，底座模型如果保持 16-bit，`65B` 模型微调仍需要超过 `780GB` 显存，远超单机研究环境的承受范围。论文在背景分析里指出，LoRA 真正的显存瓶颈并不主要来自 adapter 参数，而是来自底座模型和激活梯度，因此只缩小 adapter 规模本身并不能解决问题；同时，作者还发现如果仍只在少量 q/v 层挂 adapter，就难以恢复 16-bit 微调性能。方法部分由此提出 `QLoRA`：底座模型用 `NF4` 进行 4-bit 存储，运行时反量化到 BF16 做矩阵计算，并配合 double quantization 压缩量化常数、用 paged optimizer 处理长序列带来的内存尖峰；更重要的是，把 LoRA adapter 接到所有线性层上而不是局限在少数 attention projection。正文实验表明，`QLoRA` 可以在单张 `48GB` GPU 上微调 `65B` 模型，并在 GLUE、T5 指令微调和 LLaMA 系列实验中恢复到与 16-bit LoRA 或 full finetuning 接近的性能。

- **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models** 正文引言把焦点收缩到 LoRA 初始化本身：LoRA 虽然保留了结构上的高效性，但 `A` 高斯初始化、`B` 零初始化会使早期梯度要么为零、要么方向接近随机，导致训练在起点附近停滞较久。论文第二页的分析明确把 full fine-tuning、LoRA 与 PiSSA 放在同一框架下比较，并指出原始权重矩阵的 principal singular components 更接近真正有效的更新方向，因此不应在最重要的低秩子空间里从随机噪声开始。`PiSSA` 的方法是先对原始矩阵 `W` 做 SVD，把主奇异值与奇异向量拆成 `A`、`B` 初始化的来源，把剩余长尾部分写入冻结的残差矩阵 `W^{res}`，从而保持前向形式仍与 LoRA 一致，但训练一开始就在更有信息量的方向上更新。正文实验显示，PiSSA 在数学、代码、对话和 GLUE 等任务上都比 LoRA 收敛更快、结果更好；当结合 4-bit 量化时，QPiSSA 还能比 QLoRA 明显降低初始量化误差。

- **LoRA-GA: Low-Rank Adaptation with Gradient Approximation** 正文引言指出 LoRA 的另一个系统性问题：单步成本虽然低，但如果需要 `5-6x` 更多训练步数才能达到类似效果，总 FLOPs 未必更省。论文方法分析把原因压到“第一步更新方向是否对齐 full fine-tuning 梯度”上，并把目标明确写成让 `∆(BA)` 在初始化后尽可能逼近 `ζ∆W`。据此，`LoRA-GA` 不再从权重主成分出发，而是先对 full gradient 做 SVD，用梯度矩阵的特征方向初始化 `A` 与 `B`，并重新设定非零初始化下的 scaling，使输出方差在不同 rank 下保持稳定。正文实验中，这种做法在 T5-Base 的 GLUE 子集以及 Llama 2-7B 的 MT-Bench、GSM8K、HumanEval 上都比 vanilla LoRA 收敛更快，通常能带来 `2-4x` 的收敛提速，并把最终性能拉近到 full fine-tuning。
