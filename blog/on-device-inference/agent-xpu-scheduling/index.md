---
title: "Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC"
date: 2026-04-02T10:31:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "agent-xpu-scheduling"
tags: ["on-device", "Agent", "Scheduling"]
categories: ["paper"]
math: true
---

# Agent.xpu

- **Agent.xpu** 正文引言先把 agent workload 与普通单轮推理明确区分开：reactive flow 需要对前台用户请求做低延迟响应，proactive flow 则会在后台长期运行，两类 flow 会在 prefill、decode、工具调用以及等待外部事件之间不断切换，因此传统静态推理引擎假设的“单次、独占、无优先级”执行模型不再成立。论文随后通过 profiling 总结出三条系统性矛盾：NPU 与 iGPU 对不同 operator 的 affinity 明显不同；共享 DDR 带宽下的并发争用具有强烈不对称性，memory-bound kernel 更容易互相拖慢；而 prefill 与 decode 的 batching 效果也完全不同于云侧 serving 的连续 batching 假设。围绕这些观察，`Agent.xpu` 构建了 heterogeneous execution graph，把 operator 的 accelerator affinity、elastic binding 与性能注释写进图结构中，再在运行时通过 prefill/decode 分离的 NPU-iGPU 协调、细粒度 preemption 和 slack-aware piggybacking 处理 reactive 与 proactive 混部。正文实验显示，相比工业级 iGPU-only serving 和静态 NPU-iGPU 方案，`Agent.xpu` 能把 proactive throughput 提升到 `1.2-4.9x`，同时把 reactive latency 压低至少 `91%`，并降低额外能耗与图形干扰。
