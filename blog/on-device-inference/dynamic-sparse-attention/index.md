---
title: "Dynamic Sparse Attention on Mobile SoCs"
date: 2026-04-02T10:30:00+08:00
slug: "dynamic-sparse-attention"
tags: ["on-device", "Attention-Optimization"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

注意力机制（Attention）在长序列下的 $O(n^2)$ 复杂度是端侧推理的发热和性能天敌。
其主要贡献：
1. **内容驱动稀疏化 (Content-driven Sparcity)**: 根据输入内容动态预测注意力图中不重要的块（Blocks），并直接跳过计算。
2. **SoC 友好算子**: 专门针对手机 GPU 的计算管线，优化了稀疏矩阵乘法的寄存器分配。

## 2. 深度分析 (Analysis)

### 优势分析：
在处理 1024 以上的长文本时，Attention 的计算占比会迅速拉升。本文的动态稀疏化在保证 99% 精度保留的同时，将长序列推理的 **端到端延迟降低了 40%**。这在端侧的长文档分析和对话 Agent 中是刚需技术。
