---
title: "HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators"
date: 2026-04-02T10:32:00+08:00
slug: "heterollm-soc"
tags: ["on-device", "Heterogeneous"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

HeteroLLM 的核心是“物尽其用”。
- **协同推理**: 将 LLM 的不同层或不同算子分配到手机 SoC 上的 CPU、GPU 和 NPU/DSP。
- **自适应划分**: 动态根据实时功耗和负载，调整每个加速器的参与度。

## 2. 深度分析 (Analysis)

相比于单一处理器的方案，异构协作能更好地平摊热量分布，避免手机因局部发热过快导致的 CPU/GPU 降频。这种从系统全局视角出发的架构设计，是未来旗舰手机实现全天候 AI 助手的关键。
