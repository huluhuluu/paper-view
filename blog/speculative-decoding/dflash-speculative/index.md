---
title: "DFlash: Block Diffusion for Flash Speculative Decoding"
date: 2026-04-02T10:14:00+08:00
slug: "dflash-speculative"
tags: ["speculative-decoding", "Diffusion"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

DFlash 引入了 **扩散模型 (Diffusion)** 的思想，试图打破传统自回归草稿生成的限制。其主要贡献在于：
1. **并行块生成 (Parallel Block Generation)**: 不同于 EAGLE 的序列生成，DFlash 可以同时生成一整块（Block）token。
2. **加速比优化**: 针对云侧大批量推理任务，展示了比自回归推测更高的并行度上限。

## 2. 核心方法 (Methods)

### 2.1 扩散机制
DFlash 将未来一段序列的 token 视为待扩散的噪声。通过多步离散扩散过程，在单次计算中还原出多个连续的 token。这种非自回归（NAR）的方式极大地压缩了草稿生成阶段的时延。

### 2.2 训练细节
引入了对比学习来增强扩散模型在有限步数下的预测精度。

## 3. 深度分析与对比 (Analysis & Comparison)

### 为什么 DFlash 适合云侧？
- **并行化极致**: DFlash 可以在单次运算中预测未来 $N$ 个 token，而 EAGLE 需要 $N$ 次运算。虽然单次扩散的开销大于单次 Transformer 层，但在高并发场景下收益更明显。
- **准确性权衡**: 由于是非自回归，其预测的一致性（Coherence）略低于 EAGLE，但在特定垂直领域表现极佳。

### 结论：
DFlash 的出现意味着推测解码正在向非自回归（NAR）领域扩展。这与云侧追求极致吞吐量（Throughput）的目标完美契合。
