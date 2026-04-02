---
title: "LLM Inference at the Edge: Mobile, NPU, and GPU Benchmarks"
date: 2026-04-02T10:33:00+08:00
slug: "llm-inference-edge"
tags: ["on-device", "Benchmarks"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

这是一篇极具实战指导意义的 Benchmark 论文。
- **深度测评**: 对比了苹果 A 系列、高通骁龙系列在运行主流 LLM 时的 NPU 与 GPU 性能差异。
- **核心结论**: 在 Batch=1 的手机推理场景中，NPU 的单位功耗性能（Efficiency）通常是 GPU 的 2x 以上。

## 2. 深度分析 (Analysis)

该研究为端侧开发者提了一个醒：虽然 GPU 易于使用，但为了长期的用户体验（低功耗、长续航），投入 NPU 算子开发虽然困难，但回报极其丰厚。
