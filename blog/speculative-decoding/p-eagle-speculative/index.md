---
title: "P-EAGLE: Parallel-Drafting EAGLE with Scalable Training"
date: 2026-04-02T10:15:00+08:00
slug: "p-eagle-speculative"
tags: ["speculative-decoding"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

P-EAGLE 是对 EAGLE 工业化应用的一次重大升级。其核心贡献：
1. **并行采样架构**: 将原本顺序依赖的草稿生成步骤，通过流水线和多级头优化为准并行结构。
2. **训练可扩展性**: 极大地缩短了训练周期。

## 2. 核心方法 (Methods)

### 2.1 多级头设计 (Multi-level Heads)
P-EAGLE 借鉴了 Medusa 的多头并行，但在每个头内部保留了 EAGLE 的特征层交互。这实现了“并行预测”与“高阶依赖捕获”的平衡。

## 3. 深度分析与对比 (Analysis & Comparison)

### 核心优势分析：
EAGLE 系列最被诟病的是在 GPU 上生成草稿时仍然需要多次串行启动 Kernels。P-EAGLE 通过优化算子融合和并行采样，减少了 Kernel Launch 的开销。这标志着算法设计与工程实现的深度结合。
