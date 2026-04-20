---
title: "PowerInfer-2: Fast Large Language Model Inference on a Smartphone"
date: 2026-04-02T10:21:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "powerinfer2-smartphone"
tags: ["on-device", "Sparsity"]
categories: ["paper"]
math: true
---

# PowerInfer-2

- **PowerInfer-2** 正文引言把手机端推理的核心问题写成两条并行约束：模型既放不下，也跑不快。论文进一步指出，PC 上已经有效的稀疏推理方案直接搬到手机会明显失效，根源在于两项硬件差异：一是移动 NPU 缺少对不规则稀疏计算的支持，导致稀疏部分只能落到 CPU；二是 UFS 的随机读性能远低于 PC 上的 NVMe，使按神经元细粒度取数的代价急剧放大。基于这个分析，`PowerInfer-2` 把 neuron cluster 设为统一的计算与存储基本单元，在计算侧根据 batch size 动态变化的 sparsity pattern，把 dense cluster 分配给 NPU、cold cluster 分配给 CPU，并在 decode 阶段持续调整 CPU-NPU 比例；在存储侧则引入 hot/cold neuron cache、cluster 级 pipeline 和针对不同量化格式的差异化加载策略，尽可能把 I/O 隐藏在计算之后。正文实验显示，该系统在手机上实现了比先前框架最高 `27.8x` 的提速，是首个能在手机上运行 `47B` 级 LLM 的系统之一，并在 OnePlus 12 上达到 `11.68` tokens/s。
