---
title: "Paper阅读笔记"
date: 2026-04-01T15:30:00+08:00
lastmod: 2026-04-02T10:40:00+08:00
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
深入解析推测解码这一“空间换时间”的并行化加速路径。

| 类别 | 包含内容 | 状态 |
|------|----------|------|
| **基础框架** | Medusa (MLP Heads), EAGLE (Feature-level) | ✅ 已完成 |
| **动态树优化** | EAGLE-2, EAGLE-3 (Training-time Test) | ✅ 已完成 |
| **云侧扩展** | P-EAGLE, DFlash (Diffusion-based) | ✅ 已完成 |
| **机制探讨** | Speculative Speculative, Vocabulary Pruning | ✅ 已完成 |

---

### 2. 端侧推理优化 (On-device Inference)
聚焦于智能手机及边缘计算设备（Mobile SoCs）上的极速推理实践。

| 类别 | 包含内容 | 状态 |
|------|----------|------|
| **硬件加速** | ASPLOS'25 NPU, Dynamic Sparse Attention | ✅ 已完成 |
| **系统调度** | Agent.xpu, HeteroLLM (Heterogeneous SoC) | ✅ 已完成 |
| **内存/存储** | LLM in a flash, Scaling Test-time Compute | ✅ 已完成 |
| **稀疏性应用** | PowerInfer-2 (Smartphone Fast Inference) | ✅ 已完成 |

---

## 项目信息
- **GitHub Repo**: [huluhuluu/paper-view](https://github.com/huluhuluu/paper-view)
- **子目录说明**: 详细文章存放于 `blog/` 目录下，通过 Hugo Module 挂载。
