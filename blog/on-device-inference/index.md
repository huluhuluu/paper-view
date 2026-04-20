---
title: "端侧 LLM 推理优化 系列论文深度解析"
date: 2026-04-02T14:00:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "on-device-inference-overview"
tags: ["on-device", "LLM-Inference", "NPU", "Mobile"]
categories: ["paper"]
math: true
---

# 端侧 LLM 推理优化 系列论文深度解析

这组工作放在一起看，正文里反复出现的并不是单一的“端侧加速”，而是约束层层下沉后的系统问题：先是容量装不下，再是 I/O 跟不上，随后是 NPU 与 GPU/CPU 如何分工，最后才是 agent 场景下多条 LLM flow 的调度和 test-time compute 分配。

- **[LLM in a Flash](/p/llm-in-flash-ssd/)** 正文把起点放在一个非常硬的容量约束上：很多 LLM 的参数规模已经超过个人设备 DRAM 上限，因此问题不是“怎样把整个模型塞进内存”，而是“怎样让 flash 参与推理且不把延迟拉爆”。论文随后把 flash 读写特性写进 cost model，并沿着“减少传输量、增大读取块、降低 DRAM 管理开销”三条线提出 `windowing`、`row-column bundling` 与专门的内存管理结构，这代表端侧推理里最基础的存储层重写。

- **[PowerInfer-2](/p/powerinfer2-smartphone/)** 正文进一步指出，手机端的问题不仅是容量不够，还存在 PC 方法难以迁移的两道硬约束：移动 NPU 不擅长非结构化稀疏计算，UFS 对随机读又远慢于 PC 上的 NVMe。论文因此把计算与存储的最小单元改写为 neuron cluster，一方面把 dense cluster 映射到 NPU、sparse cluster 映射到 CPU，另一方面用 cluster 级 pipeline、cache 和按需加载来隐藏 I/O 延迟，把端侧稀疏推理真正落到手机硬件上。

- **[HeteroLLM](/p/heterollm-soc/)** 这篇工作从异构 SoC 的基础特性出发重写执行引擎。正文分析显示，移动 NPU 的效率强烈依赖 tensor order、shape 和 stage，而 GPU 与 NPU 共享统一内存空间且单独一类处理器无法吃满整体内存带宽。基于这些观察，论文提出 `HeteroInfer`，把 layer-level、weight-centric、activation-centric 与 hybrid partition 组合起来，并配合基于 unified memory 的 fast synchronization，在 prefill 与 decode 两个阶段分别设计不同的 GPU-NPU 并行方式。

- **[LLM Inference at the Edge](/p/llm-inference-edge/)** 这篇论文没有继续堆新算子，而是直接用持续负载 benchmark 去回答“平台真实上限在哪里”。正文把 warm-condition 下的连续推理作为实验对象，结果显示不同平台的主瓶颈完全不同：手机主要受 thermal management 支配，RTX 4050 受 battery power ceiling 限制，而 Hailo-10H 更接近 memory-bandwidth-bound。它的重要性在于把端侧部署的判断标准从峰值 TOPS 拉回到了 sustained throughput、power 与 thermal behavior。

- **[Agent.xpu](/p/agent-xpu-scheduling/)** 如果说前面的工作还主要面对单次推理，这篇论文则把问题推进到了 personal agent 的并发 flow。正文指出，reactive flow 与 proactive flow 会在 prefill、decode、工具调用和等待事件之间不断切换，而现有 on-device engine 仍假设静态、单次、无优先级的推理过程。围绕这个矛盾，论文提出异构执行图 `HEG`、prefill/decode 分离的 NPU-iGPU 协调、细粒度抢占和 slack-aware piggybacking，使 SoC 能同时照顾 reactive 延迟和 proactive 吞吐。

- **[Scaling Test-time Compute for LLM Agents](/p/scaling-test-time/)** 这篇论文把端侧或本地 agent 再往上一层推进到 test-time compute。正文分析认为，agent 任务不是一次性解题，而是多步决策与环境交互，因此错误会沿长链路传播，简单复制数学推理里的 `best-of-N` 并不稳。论文围绕 parallel sampling、sequential revision、verifier 与 merge method 展开系统实验，发现 agent 场景依然存在 test-time scaling，但关键不只是“多采样”，而是“什么时候反思”以及“如何在多条 rollout 中做更可靠的比较”。
