---
title: "HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators"
date: 2026-04-02T10:32:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "heterollm-soc"
tags: ["on-device", "Heterogeneous"]
categories: ["paper"]
math: true
---

# HeteroLLM

- **HeteroLLM** 正文引言首先指出，现有移动端推理引擎大多只绑定单一 accelerator，因此即使 SoC 同时具备 GPU 与 NPU，也难以把两者的计算能力和共享内存带宽一起利用起来。论文随后的表征分析给出三条更细的结论：第一，NPU 的性能对 tensor order、shape 以及 prefill/decode 所处阶段都高度敏感，并不是所有矩阵都适合直接丢给 NPU；第二，统一内存架构让 GPU-NPU 之间具备快速同步的可能；第三，单一处理器在 decoding 这样的 memory-bound 场景下并不能吃满整颗 SoC 的内存带宽。基于这些观察，`HeteroInfer` 在方法上同时采用 layer-level 与 tensor-level 的异构切分策略，包括 weight-centric、activation-centric 与 hybrid partition，并进一步用可预测等待时间上的 fast synchronization 来缩短 GPU-NPU 交接延迟。正文实验表明，在 Galaxy S24 Ultra 上，这套异构执行相较 GPU-only 或 NPU-only 路线都能取得显著加速，在高精度计算下 prefill 超过 `1000` tok/s、decode 超过 `50` tok/s，同时对其他图形应用的干扰保持较低。
