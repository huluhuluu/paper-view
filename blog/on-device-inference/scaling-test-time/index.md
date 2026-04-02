---
title: "Scaling Test-time Compute for LLM Agents"
date: 2026-04-02T10:34:00+08:00
slug: "scaling-test-time"
tags: ["on-device", "Agent", "Test-time-Compute"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

如何在 4-bit 量化的 7B 端侧模型上实现接近 70B 模型在复杂任务上的表现？
- **策略**: 通过多轮迭代、验证和 Self-Consistency，让模型在推理时多花一些“思考时间”。
- **方法**: 在端侧实现了动态 Beam Search 和评估器（Evaluator）机制。

## 2. 深度分析 (Analysis)

这是对“模型规模”论的一种补充。在端侧，如果任务不要求即时响应（比如生成一段分析摘要），通过扩展 Test-time Compute，小模型能爆发惊人的能力。这为端侧小模型的智能化提升提供了另一条路径。
