---
title: "shadowAttn: Dynamic Sparse Attention on Mobile SoCs"
date: 2026-04-02T10:30:00+08:00
slug: "dynamic-sparse-attention"
tags: ["on-device", "Attention-Optimization", "NPU"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

注意力机制（Attention）在长序列下的 $O(n^2)$ 复杂度是端侧推理的发热和性能天敌。本文提出 **shadowAttn**，主要贡献：

1. **NPU-centric 稀疏注意力**: 通过将重要 token 的估算阶段卸载到 NPU，最小化对 CPU/GPU 的依赖。
2. **系统-算法协同设计**: 提出 NPU 计算图分桶（Bucketing）、Head-wise NPU-CPU/GPU 流水线、Per-head 细粒度稀疏比等技术。
3. **量化敏感性利用**: 发现确定重要 token 只需要注意力分数的相对值，而非精确值，因此估算阶段对量化更鲁棒。

## 2. 核心方法 (Methods)

### 2.1 问题背景

在现代端侧推理框架中，Attention 操作经常从 NPU 回退到 CPU/GPU，原因在于量化敏感性：
- LLM 的激活值比权重更难量化
- NPU 的静态图机制限制了细粒度量化

实验发现 NPU-based attention 在移动 LLM 任务上平均导致 18pp 的精度下降。

### 2.2 shadowAttn 设计

**核心洞察**: 在 Qwen2-1.5B 模型上，WikiText-2 数据集中平均超过 80% 的 token 被分配较低的注意力权重。

**关键创新**:
- **NPU-based Pilot Compute**: 在 NPU 上执行重要 token 的估算
- **位置索引传递**: 仅将重要 token 的位置索引传给 CPU/GPU 进行稀疏计算
- **Head-wise 稀疏比**: 每个 attention head 使用不同的稀疏比，通过轻量级离线分析确定

## 3. 深度分析 (Analysis)

### 优势分析：

- **资源效率**: 仅需 1 个 CPU 核心，其他计算完全在 NPU 上
- **性能提升**: 相比现有框架高达 6.9× 加速，4.5× 端到端加速
- **能耗降低**: 7.7× 能耗减少
- **精度保持**: 平均精度损失仅 0.4 pp

### 更高层次分析：

shadowAttn 解决了 NPU-centric LLM 推理的关键问题。传统稀疏注意力方法在估算阶段的开销占总计算时间的 60% 以上（在 80% 稀疏度时）。通过将估算卸载到 NPU，shadowAttn 实现了真正的端侧高效推理。

这标志着端侧注意力优化从"如何算得更快"转向"如何更智能地跳过不重要的计算"。
