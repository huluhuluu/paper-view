---
title: "LLM 推理系统优化深度解析"
date: 2026-04-02T17:10:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "llm-inference-optimization"
tags: ["inference", "FlashAttention", "PagedAttention", "vLLM"]
categories: ["paper"]
math: true
---

# LLM 推理系统优化深度解析

这些工作分别把推理系统的瓶颈拆到了不同层面：`Roofline` 先给出判断瓶颈的位置，`FlashAttention` 把 attention 的代价改写成 IO 问题，`PagedAttention` 把吞吐上限落到 KV cache 管理，而 `Orca` 则把服务侧低效归因到错误的调度粒度。

- **Roofline: An Insightful Visual Performance Model for Multicore Architectures** `Roofline` 正文讨论的不是某个具体模型，而是多核体系结构上的 kernel 为什么常常达不到理论峰值。作者把性能上界同时写成算力峰值与内存带宽上界的最小值，并用 operational intensity 作为连接两者的核心量：当程序的每字节内存访问能支撑的计算太少时，瓶颈不在 FLOPs，而在数据搬运；只有越过 ridge point，才会真正进入 compute-bound 区域。这个框架的重要性在于，它把后续的优化方向从“继续堆算子实现”改成了先判断瓶颈究竟在 HBM/DRAM 还是在算力单元，这一点后来在 attention 和 serving 系统里都成为基础分析工具。

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** 正文引言明确指出，长序列 attention 慢并不只是因为二次复杂度，而是标准实现会显式物化 `N×N` 的中间矩阵 `S` 和 `P`，从而在 HBM 与片上 SRAM 之间反复搬运数据。论文第二页进一步批评许多近似 attention 方法只减少 FLOPs 却不减少 IO，因此 wall-clock speedup 并不明显。为了解决这个问题，`FlashAttention` 把前向过程改写成按块加载 `Q/K/V` 的 tiled 计算，在片上完成增量 softmax 统计 `m` 与 `l`，避免把整张 attention matrix 写回 HBM；反向则通过保存归一化统计量、重新计算局部 attention，换掉对大中间矩阵的读写。正文实验显示，这种 IO-aware 设计在 GPT-2 上可带来约 `3x` 训练提速，在 BERT-large 上超过当时 MLPerf 纪录，并把 attention 的额外显存从二次规模压到线性规模。

- **Efficient Memory Management for Large Language Model Serving with PagedAttention** 这篇论文正文把 serving 吞吐的关键从“单个 kernel 多快”转移到“KV cache 怎么分配”。作者分析发现，现有系统把一条序列的 KV cache 预留在连续显存里，会同时引入 reservation、internal fragmentation 和 external fragmentation，真实 token state 只占到总 KV cache 空间的一小部分。`PagedAttention` 因此借用了操作系统里的虚拟内存思想，把序列拆成固定大小的逻辑块，再通过 block table 映射到物理块，使 KV cache 可以按需增长而不要求物理连续；在并行采样和 beam search 场景下，还能通过 block-level sharing 和 copy-on-write 共享前缀。正文实验显示，`vLLM` 基于这套设计能把 KV cache 浪费压到接近零，并在延迟相近条件下比 FasterTransformer 和 Orca 取得 `2-4x` 吞吐提升，长序列和复杂解码下收益更明显。

- **Orca: A Distributed Serving System for Transformer-Based Generative Models** `Orca` 正文把生成式 serving 的核心矛盾写成“请求是多轮迭代推进的，但系统却按整条请求静态调度”。在传统 request-level batching 下，一批请求一旦进入执行引擎就不能中途插入、退出或返回结果，较短请求会被长请求拖住，新到请求也必须等待整个 batch 完结。为了解决这个问题，`Orca` 提出 iteration-level scheduling，让 serving system 每次只调度单轮迭代；同时为了兼容不同请求处于不同 token 位置的情况，正文又设计了 selective batching，只对适合批处理的操作做 batching，而对 attention 这类 shape 随 token 累积变化的部分按请求分别执行。最终，论文在 GPT-3 175B serving 上报告了相对 FasterTransformer 在同等延迟下最高 `36.9x` 的吞吐提升，说明生成式模型的服务效率很大程度上取决于调度方式，而不只是 kernel 本身。
