---
title: "Paper阅读笔记"
date: 2026-04-01T15:30:00+08:00
lastmod: 2026-04-02T12:00:00+08:00
draft: false
description: "端侧AI系统优化、LLM推理加速、推测解码等论文深度解析"
slug: "paper-view"
tags: ["paper"]
categories: ["paper"]
comments: true
math: true
---

# Paper 阅读笔记中心

本板块记录了关于 LLM 推理优化、端侧部署以及推测解码（Speculative Decoding）等核心领域的论文研究。本专栏旨在通过深度剖析，揭示模型在异构 SoC（NPU/GPU）上的运行机理与优化方向。

---

## 专题大纲

### 1. 推测解码 (Speculative Decoding)
深入解析推测解码这一"空间换时间"的并行化加速路径。

**完整综述**: [推测解码系列论文深度解析](../speculative-decoding/) - 从 MEDUSA 到 DFlash，系统性梳理推测解码技术演进

| 论文 | 核心贡献 | 详解链接 |
|------|----------|----------|
| MEDUSA | 多头并行预测，无需独立草稿模型 | [详见综述](../speculative-decoding/#一基础框架medusa-与多头并行预测) |
| EAGLE | 特征层自回归，重塑不确定性 | [详见综述](../speculative-decoding/#二特征层自回归eagle-系列的范式转移) |
| EAGLE-2 | 动态草稿树，上下文自适应 | [详见综述](../speculative-decoding/#22-eagle-2-faster-inference-of-language-models-with-dynamic-draft-trees) |
| EAGLE-3 | Training-Time Test，Scaling Law | [详见综述](../speculative-decoding/#23-eagle-3-scaling-up-inference-acceleration-via-training-time-test) |
| DFlash | 扩散模型并行块生成 | [详见综述](../speculative-decoding/#41-dflash-block-diffusion-for-flash-speculative-decoding) |
| P-EAGLE | 并行草稿，可扩展训练 | [详见综述](../speculative-decoding/#42-p-eagle-parallel-drafting-eagle-with-scalable-training) |
| SSD | 双重推测，异步流水线 | [详见综述](../speculative-decoding/#43-speculative-speculative-decoding-ssd) |
| SpecVocab | 词表推测，上下文感知 | [详见综述](../speculative-decoding/#61-speculative-decoding-with-a-speculative-vocabulary-specvocab) |
| SVIP | 自验证动态长度策略 | [详见综述](../speculative-decoding/#62-draft-model-knows-when-to-stop-self-verification-for-long-form-generation) |

---

### 2. 端侧推理优化 (On-device Inference)
聚焦于智能手机及边缘计算设备（Mobile SoCs）上的极速推理实践。

| 论文 | 核心贡献 | 详解链接 |
|------|----------|----------|
| [ASPLOS'25] Fast On-device LLM Inference with NPUs | NPU 指令级融合，20+ tokens/s | [详细解析](asplos25-npu-llm/) |
| shadowAttn: Dynamic Sparse Attention | NPU-centric 稀疏注意力，6.9× 加速 | [详细解析](dynamic-sparse-attention/) |
| Agent.xpu | Agentic LLM 流调度，异构协同 | [详细解析](agent-xpu-scheduling/) |
| HeteroLLM | 层级/张量级异构执行，9.99× 加速 | [详细解析](heterollm-soc/) |
| LLM in a Flash | 闪存为缓存，突破内存限制 | [详细解析](llm-in-flash-ssd/) |
| LLM Inference at the Edge | NPU vs GPU Benchmark | [详细解析](llm-inference-edge/) |
| Scaling Test-time Compute | Test-time 扩展，小模型大能力 | [详细解析](scaling-test-time/) |
| PowerInfer-2 | 稀疏性利用，动态权重加载 | [详细解析](powerinfer2-smartphone/) |

---

## 项目信息
- **GitHub Repo**: [huluhuluu/paper-view](https://github.com/huluhuluu/paper-view)
- **子目录说明**: 详细文章存放于 `blog/` 目录下，通过 Hugo Module 挂载。