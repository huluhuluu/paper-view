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

# Paper 阅读笔记

本板块记录了关于 LLM 推理优化、端侧部署以及推测解码（Speculative Decoding）等核心领域的论文研究。

---

## 1. 推测解码 (Speculative Decoding)

推测解码是"空间换时间"的并行化加速路径，使用轻量级草稿模型快速生成候选 token，然后由目标模型并行验证。

| 论文 | 核心贡献 | 链接 |
|------|----------|------|
| MEDUSA | 多头并行预测，无需独立草稿模型 | [传送门](/p/speculative-decoding-overview/) |
| EAGLE | 特征层自回归，重塑不确定性 | [传送门](/p/speculative-decoding-overview/) |
| EAGLE-2 | 动态草稿树，上下文自适应 | [传送门](/p/speculative-decoding-overview/) |
| EAGLE-3 | Training-Time Test，Scaling Law | [传送门](/p/speculative-decoding-overview/) |
| DFlash | 扩散模型并行块生成 | [传送门](/p/speculative-decoding-overview/) |
| P-EAGLE | 并行草稿，可扩展训练 | [传送门](/p/speculative-decoding-overview/) |
| SSD | 双重推测，异步流水线 | [传送门](/p/speculative-decoding-overview/) |
| SpecVocab | 词表推测，上下文感知 | [传送门](/p/speculative-decoding-overview/) |
| SVIP | 自验证动态长度策略 | [传送门](/p/speculative-decoding-overview/) |

---

## 2. 端侧推理优化 (On-device Inference)

聚焦于智能手机及边缘计算设备（Mobile SoCs）上的极速推理实践。

| 论文 | 核心贡献 | 链接 |
|------|----------|------|
| Fast On-device LLM Inference with NPUs | NPU 指令级融合，20+ tokens/s | [传送门](/p/asplos25-npu-llm/) |
| shadowAttn: Dynamic Sparse Attention | NPU-centric 稀疏注意力，6.9× 加速 | [传送门](/p/dynamic-sparse-attention/) |
| Agent.xpu | Agentic LLM 流调度，异构协同 | [传送门](/p/agent-xpu-scheduling/) |
| HeteroLLM | 层级/张量级异构执行，9.99× 加速 | [传送门](/p/heterollm-soc/) |
| LLM in a Flash | 闪存为缓存，突破内存限制 | [传送门](/p/llm-in-flash-ssd/) |
| LLM Inference at the Edge | NPU vs GPU Benchmark | [传送门](/p/llm-inference-edge/) |
| Scaling Test-time Compute | Test-time 扩展，小模型大能力 | [传送门](/p/scaling-test-time/) |
| PowerInfer-2 | 稀疏性利用，动态权重加载 | [传送门](/p/powerinfer2-smartphone/) |

---

## 相关链接

- **GitHub Repo**: [huluhuluu/paper-view](https://github.com/huluhuluu/paper-view)
