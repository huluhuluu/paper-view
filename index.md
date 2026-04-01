---
title: "论文阅读笔记"
date: 2026-04-01T15:30:00+08:00
lastmod: 2026-04-01T15:30:00+08:00
draft: false
description: "端侧AI、LLM推理加速、小模型等论文阅读笔记"
slug: "paper-view"
tags: ["论文", "端侧AI", "LLM推理"]
categories: ["论文阅读"]
comments: true
math: true
---

# 论文阅读笔记

端侧AI系统优化、LLM推理加速、小模型增强等方向的论文阅读笔记。

## 目录

### 端侧系统优化

| 论文 | 说明 | 状态 |
|------|------|------|
| [LLM in a Flash](/p/llm-in-flash/) | 有限内存下的高效 LLM 推理 | 📝 TODO |
| [PowerInfer-2](/p/powerinfer2/) | 手机端快速 LLM 推理 | 📝 TODO |
| [MNN-LLM](/p/mnn-llm/) | 移动端 LLM 通用推理引擎 | 📝 TODO |
| [HeteroLLM](/p/heterollm/) | 移动 SoC 异构 AI 加速器 | 📝 TODO |
| [EdgeShard](/p/edgeshard/) | 协作边缘计算 LLM 推理 | 📝 TODO |

### 小模型能力增强

| 论文 | 说明 | 状态 |
|------|------|------|
| [Scaling Test-time Compute](/p/scaling-testtime/) | LLM Agent 测试时计算扩展 | 📝 TODO |
| [Inference Scaling Laws](/p/inference-scaling/) | 问题求解的计算最优推理 | 📝 TODO |
| [Small Language Models Survey](/p/slm-survey/) | 小语言模型综述 | 📝 TODO |

### LLM 推理加速

| 论文 | 说明 | 状态 |
|------|------|------|
| [MIRROR Speculative Decoding](/p/mirror-speculative/) | 打破串行障碍的推测解码 | 📝 TODO |
| [InfiniPot](/p/infinipot/) | 内存受限下的无限上下文处理 | 📝 TODO |
| [Scaling On-Device GPU](/p/scaling-gpu/) | 大生成模型的端侧 GPU 推理 | 📝 TODO |
| [LLM Inference at the Edge](/p/llm-edge/) | 移动端 NPU/GPU LLM 推理 | 📝 TODO |

---

## 研究方向

### 端侧系统优化

关注移动设备、边缘计算场景下的 LLM 部署优化，包括：
- 内存优化（Flash Attention、KV Cache）
- 异构计算（CPU/GPU/NPU/DSP）
- 模型压缩与量化

### 小模型能力增强

通过测试时计算扩展提升小模型能力：
- Chain-of-Thought (CoT)
- Beam Search
- Self-Consistency

### LLM 推理加速

推理阶段的加速技术：
- Speculative Decoding（推测解码）
- Continuous Batching
- PagedAttention / FlashAttention
- KV Cache 优化