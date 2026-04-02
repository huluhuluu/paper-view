---
title: "[ASPLOS 2025] Fast On-device LLM Inference with NPUs"
date: 2026-04-02T10:20:00+08:00
slug: "asplos25-npu-llm"
tags: ["on-device", "NPU", "ASPLOS"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

ASPLOS 2025 的这篇工作将手机端 LLM 推理推向了新的高度。其核心贡献：
1. **指令级算子融合 (Instruction-level Fusion)**: 针对移动端 NPU 的底层指令集，将内存密集型的算子直接融合进计算指令流，显著减少了访存开销。
2. **混合精度与非对称量化**: 在 NPU 的 INT4 和 FP16 精度间进行最优分配，保证模型精度的同时，利用 NPU 的专用加速单元。
3. **内存重用 (Memory Reuse)**: 极大提升了在极窄带宽（Bandwidth-bound）下的数据搬运效率。

## 2. 核心方法 (Methods)

### 2.1 针对 NPU 的深度定制
NPU 不同于通用的 GPU，其算力强但生态碎。本文通过自定义指令流水线，实现了 Llama-3-8B 在手机 NPU 上以 **20+ tokens/s** 的速度流畅运行。

### 2.2 软硬件协同
通过量化感知训练（QAT）和 NPU 的本地调度器配合，消除了 CPU 到 NPU 之间的任务调度延迟。

## 3. 深度分析与更高层次分析 (Analysis)

### 为什么这篇文章很重要？
在移动端，GPU 的主要职责是渲染 UI 和图形任务，长时间高负荷运行 LLM 会导致严重的“发热降频”。
- **优势**: NPU 作为专门的 AI 加速芯片，功耗极低。ASPLOS'25 的工作证明了：**端侧 LLM 的未来必然是 NPU-First**。
- **行业趋势**: 这篇论文预示着，随着 Agent 类应用在手机端的常驻（Always-on），只有通过底层算子与硬件指令的高度适配，才能在保证续航的前提下，提供秒级的端侧大模型响应。
