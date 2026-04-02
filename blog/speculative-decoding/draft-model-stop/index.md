---
title: "Draft Model Knows When to Stop: Self-Verification for Long-Form Generation"
date: 2026-04-02T10:18:00+08:00
slug: "draft-model-stop"
tags: ["speculative-decoding", "Long-Form-Generation"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

长文本生成中，推测解码常因累积不确定性导致验证失败，反而浪费算力。
其贡献在于：
1. **自感知停止机制**: 草稿模型在预测过程中，如果发现熵（Entropy）过高，会主动终止推测，提前交由大模型验证。
2. **自校准采样**: 根据上下文长度动态调整采样阈值。

## 2. 深度分析 (Analysis)

在 LLM 进入“长文本时代”后，固定长度的推测（如固定推测 5 步）不再是最佳策略。本文的核心贡献在于让草稿模型具备了“自我意识”，从而在保证生成质量的同时，最大化了加速效率。这在端侧的长文档分析和 Agent 任务中至关重要。
