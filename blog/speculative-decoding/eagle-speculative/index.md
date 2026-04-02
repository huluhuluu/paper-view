---
title: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
date: 2026-04-02T10:11:00+08:00
slug: "eagle-speculative"
tags: ["speculative-decoding", "LLM-Acceleration"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

EAGLE 在推测解码领域引入了范式转移。其核心贡献在于：
1. **特征层自回归 (Feature-level Autoregressive)**: 提出不在词表层（Token Level）做推测，而是在特征层（Hidden States Level）进行自回归推测。
2. **重塑不确定性**: 论证了特征层的不确定性远低于词表层，从而能实现更高的接受率（Acceptance Rate）。

## 2. 核心方法 (Methods)

### 2.1 结构设计
EAGLE 包含一个极轻量级的自回归组件（通常只有 1 层 Transformer）。其输入是当前 token 的嵌入向量 $e_t$ 和前一层的隐藏状态 $h_{t-1}$。
预测下一个隐藏状态：
$$h_t = \text{AutoRegressor}(e_t, h_{t-1})$$

### 2.2 工作流程
1. **Drafting**: 草稿模型在特征层快速自回归产生序列。
2. **Sampling**: 通过主模型的 LM Head 将预测的隐藏状态映射回 token。
3. **Verification**: 主模型并行验证。

## 3. 深度分析与对比 (Analysis & Comparison)

### 为什么 EAGLE 比 Medusa 强？
Medusa 的每个头是并行的、相互独立的。而 EAGLE 是自回归的。由于采用了特征层作为输入，EAGLE 能够捕获到更丰富的上下文信息。
- **数据证明**: 在 Llama-2-70B 上，EAGLE 的接受率通常能达到 80% 以上，而 Medusa 在长序列预测时准确率会骤降。
- **计算开销**: 虽然 EAGLE 引入了自回归，但因为操作的是 Hidden States 而非完整的词表，其单步延迟极低。

### 核心优势总结：
EAGLE 实际上是在模仿大模型的“思维路径”。通过在更稳定的高维特征空间进行推测，它极大地缓解了自回归生成中的“累积误差”问题。
