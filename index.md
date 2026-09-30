---
title: "Paper 阅读笔记"
date: 2026-04-01T15:30:00+08:00
lastmod: 2026-07-13T12:00:00+08:00
draft: false
description: "围绕厂商模型论文、推测解码、端侧推理、量化、LoRA 和 test-time scaling 的论文阅读笔记"
slug: "paper-view"
tags: ["paper"]
categories: ["paper"]
build:
  list: never
comments: true
math: true
---

# Paper 阅读笔记

这里整理的是我当前重点关注的几条路线：厂商模型论文、推测解码、端侧推理、量化、高效微调和推理时扩展。整体写法尽量统一成“论文要解决什么 -> 核心结构/方法 -> 我自己的判断”，方便后续继续加图和补实验。

- ✅ **已完成**：内容与结构已经基本稳定，可直接展示。

- 🚧 **TODO**：已有初稿，但仍在继续补图、补实验或调整结构。


## 1. 厂商模型论文

| 文章 | 主要内容 | 状态 | 链接 |
|------|----------|------|------|
| 厂商模型论文路线 | DeepSeek-V3/V4/R1、Qwen3、Llama 3、Gemma 2、Mixtral、Kimi K2 等模型技术报告 | 🚧 TODO | [传送门](/p/vendor-model-papers/) |

## 2. 推测解码

| 文章 | 主要内容 | 状态 | 链接 |
|------|----------|------|------|
| 推测解码主线 | Lookahead、MEDUSA、EAGLE、EAGLE-2、EAGLE-3，以及后续并行化路线 | 🚧 TODO | [传送门](/p/speculative-decoding-overview/) |
| 草稿模型训练 | Scaling Law、小模型实证、MAGICDEC、Hidden State SD、SSD | ✅ 已完成 | [传送门](/p/draft-train/) |

## 3. 端侧推理

| 文章 | 主要内容 | 状态 | 链接 |
|------|----------|------|------|
| 端侧推理总览 | 端侧 LLM 的存储、稀疏、NPU、调度与 benchmark 视角 | ✅ 完成 | [传送门](/post/on-device-inference/) |
| LLM in a Flash | 闪存作为权重缓存，解决“模型放不下” | ✅ 完成 | [传送门](/p/llm-in-flash-ssd/) |
| PowerInfer-2 | 神经元稀疏 + 动态权重加载 | ✅ 完成 | [传送门](/p/powerinfer2-smartphone/) |
| Fast On-device LLM Inference with NPUs | NPU-first 路线与指令级融合 | ✅ 已完成 | [传送门](/p/asplos25-npu-llm/) |
| shadowAttn | 面向 Mobile SoC 的动态稀疏注意力 | ✅ 完成 | [传送门](/p/dynamic-sparse-attention/) |
| MobileQuant | 面向移动 NPU/DSP 的 W/A integer-only 量化 | ✅ 完成 | [传送门](/p/mobilequant/) |
| HeteroLLM | CPU/GPU/NPU 异构切分与协同 | ✅ 完成 | [传送门](/p/heterollm-soc/) |
| Agent.xpu | Agent 工作负载的 SoC 调度 | ✅ 完成 | [传送门](/p/agent-xpu-scheduling/) |
| LLM Inference at the Edge | NPU 与 GPU 的移动端 benchmark | ✅ 完成 | [传送门](/p/llm-inference-edge/) |
| Scaling Test-time Compute for LLM Agents | 端侧小模型靠推理预算补能力 | 🚧 TODO | [传送门](/p/scaling-test-time/) |

## 4. 量化

| 文章 | 主要内容 | 状态 | 链接 |
|------|----------|------|------|
| LLM 量化技术 | `LLM.int8()`、`SmoothQuant`、`AWQ` 的主线比较 | 🚧 TODO | [传送门](/p/llm-quantization/) |

## 5. 高效微调

| 文章 | 主要内容 | 状态 | 链接 |
|------|----------|------|------|
| LoRA 微调路线 | `LoRA`、`QLoRA`、`PISSA`、`LoRA-GA` | 🚧 TODO | [传送门](/p/lora-finetuning/) |

## 6. Test-time Scaling

| 文章 | 主要内容 | 状态 | 链接 |
|------|----------|------|------|
| Test-time Scaling | `Scaling LLM Test-Time Compute Optimally` 与 `Inference Scaling Laws` | 🚧 TODO | [传送门](/p/test-time-scaling/) |

## 7. 推理系统

| 文章 | 主要内容 | 状态 | 链接 |
|------|----------|------|------|
| 推理系统优化 | `Roofline`、`FlashAttention`、`PagedAttention`、`Continuous Batching` | 🚧 TODO | [传送门](/p/llm-inference-optimization/) |
