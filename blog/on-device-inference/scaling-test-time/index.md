---
title: "Scaling Test-time Compute for LLM Agents"
date: 2026-04-02T10:34:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "scaling-test-time"
tags: ["on-device", "Agent", "Test-time-Compute"]
categories: ["paper"]
math: true
---

# Scaling Test-time Compute for LLM Agents

- **Scaling Test-time Compute for LLM Agents** 正文引言提出的问题不是“agent 能不能采样更多次”，而是“传统 test-time scaling 方法迁移到多步 agent 系统后还是否有效”。论文指出，agent 任务要把复杂问题拆成多个连续步骤，并且常常伴随工具调用与环境交互，因此每一步的误差都会往后累计；如果直接在每一步套用大规模 `best-of-N`，不仅开销巨大，还可能把错误放大到整条轨迹。基于这个判断，正文系统比较了 parallel sampling、sequential revision、verifier/result merging 以及 diversifying rollouts 等多条路线，并专门分析 reflection 的触发时机。实验部分的结果比较明确：agent 场景同样存在 test-time scaling，但不同实现之间差异很大，其中 BoN 在平行采样里最稳定，list-wise 方法在 verifier 和结果合并两端都优于直接 scoring，而 sequential revision 的关键不在于“每一步都反思”，而在于知道什么时候该反思。
