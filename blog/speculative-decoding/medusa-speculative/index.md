---
title: "MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
date: 2026-04-02T10:10:00+08:00
slug: "medusa-speculative"
tags: ["speculative-decoding", "LLM-Inference"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

MEDUSA 针对大语言模型（LLM）自回归生成的串行瓶颈，提出了一种简洁而高效的加速框架。其主要贡献在于：
1. **多头结构 (Multiple Decoding Heads)**: 不同于传统的 Speculative Decoding 需要维护一个独立的草稿模型（Draft Model），MEDUSA 直接在主模型的最后一个隐藏层上并行挂载了多个 MLP 头。
2. **无需草稿模型**: 极大地降低了系统复杂度，避免了模型间同步和草稿模型占用的额外显存。
3. **Tree Attention**: 提出了一种树状注意力机制，允许在一次大模型的前向传递中验证多个候选 token 路径。

## 2. 核心方法 (Methods)

### 2.1 Medusa Heads
在 Llama 等主模型之上，MEDUSA 挂载了 $k$ 个独立的 MLP 头。每个头负责预测未来第 $i$ 个位置（$1 \le i \le k$）的 token 分布。预测公式如下：
$$h^{(i)}_t = \text{MedusaHead}_i(s_t)$$
其中 $s_t$ 是主模型第 $t$ 步的隐藏状态。

### 2.2 Tree-based Verification
MEDUSA 并不简单地只预测一条路径，而是生成多个候选并构建出一棵“草稿树”。通过构建一个特殊的 $N \times N$ 的 Attention Mask，MEDUSA 可以在单次推理中并行验证树上的所有节点（paths）。

### 2.3 训练策略
MEDUSA 采用“冻结主模型，训练头”的策略（Freeze Backbone, Train Heads）。这种方式不仅保护了主模型的生成能力，还能在极小的数据集（如 ShareGPT）上快速收敛。

## 3. 深度分析与对比 (Analysis & Comparison)

### 优势分析：
- **延迟极低**: 相比传统的 Speculative Decoding（如 Leviathan 等），MEDUSA 避免了草稿模型生成 token 时的等待时间。
- **内存友好**: 只有一个模型，KV Cache 管理更加直接。

### 局限性与后续启发：
相比后续的 **EAGLE** 系列，MEDUSA 的主要缺陷在于其预测是“静态”的。由于每个 Medusa Head 仅基于当前的隐藏状态 $s_t$ 进行独立预测，忽略了预测过程中 token 间的相互依赖关系。这导致当推测步数 $k$ 增大时，准确率会呈指数级下降。EAGLE 正是针对这一点，将预测从词表层移回了特征层。
