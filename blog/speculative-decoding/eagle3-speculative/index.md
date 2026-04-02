---
title: "EAGLE-3: Scaling up Inference Acceleration via Training-Time Test"
date: 2026-04-02T10:13:00+08:00
slug: "eagle3-speculative"
tags: ["speculative-decoding"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

EAGLE-3 是该系列的集大成者，核心贡献集中在两个方面：
1. **训练时测试 (Training-Time Test)**: 优化了训练 Loss，使其更加贴合推理阶段的加速目标。
2. **32k 词表剪枝 (Vocabulary Pruning)**: 彻底解决了大词表（如 Llama-3 的 128k 词表）在草稿模型采样时的巨大开销。

## 2. 核心方法 (Methods)

### 2.1 词表压缩
Llama-3 的 128k 词表导致 LM Head 在采样时的 Softmax 计算极其缓慢。EAGLE-3 通过数据驱动的方法，识别并保留了最核心的 32,768 个 token。这不仅减小了模型大小，还将 Softmax 的延迟降低了 4 倍。

### 2.2 结构微调
优化了 Transformer 层的注意力分布，提升了特征还原的精度。

## 3. 深度分析与对比 (Analysis & Comparison)

### 工业价值分析：
EAGLE-3 是目前唯一在 Llama-3 128k 词表场景下依然能保持超高加速比的方案。
- **优势**: 32k 词表的设计非常巧妙。实验表明，草稿模型即使只见过 32k 个核心词，其特征预测能力依然足以引导大模型在 128k 全词表上进行验证。
- **瓶颈突破**: 当推测解码进入“快到极致”的阶段，采样和逻辑控制的开销变成了大头。EAGLE-3 正是从这些“细枝末节”中榨取了最后的性能上限。
