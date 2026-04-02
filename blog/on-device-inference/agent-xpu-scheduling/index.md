---
title: "Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC"
date: 2026-04-02T10:31:00+08:00
slug: "agent-xpu-scheduling"
tags: ["on-device", "Agent", "Scheduling"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

随着 LLM 从单一对话转向 Agentic Workflow（工具调用、多步搜索），推理负载变得更加复杂。
其核心贡献在于：
1. **任务感知调度 (Task-aware Scheduling)**: 区分“逻辑分支（CPU密集型）”与“神经推理（NPU/GPU密集型）”任务。
2. **动态平衡**: 在多任务并发时（如边录音边生成），确保 Agent 的响应延迟（Latency）不发生抖动。

## 2. 深度分析 (Analysis)

### 未来趋势：
Agent.xpu 预示着端侧 AI 的下半场：**系统级资源管理**。当手机后台运行着多个小模型（用于语音识别、视觉分析）和一个大模型（用于决策）时，如何分摊 SoC 资源将是系统工程师面临的最大难题。Agent.xpu 的调度算法为这种复杂的异构协同提供了第一份蓝图。
