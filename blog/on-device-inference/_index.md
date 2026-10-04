---
title: "端侧 LLM 推理优化"
date: 2026-04-02T14:00:00+08:00
lastmod: 2026-07-13T12:00:00+08:00
draft: false
description: "端侧 LLM 的存储、NPU、量化、稀疏、异构 SoC 与 agent 工作负载论文索引"
slug: "on-device-inference-overview"
tags: ["on-device", "LLM-Inference", "NPU", "Mobile"]
categories: ["paper"]
---

# 端侧 LLM 推理优化

端侧 LLM 推理的难点主要来自资源边界，不来自单个慢算子。四类约束最常见：模型权重和 `KV cache` 容量，`flash/DRAM` 带宽，移动 `NPU/GPU/CPU` 的 shape 与 kernel 限制，持续运行时的功耗和温度。

这一组笔记按瓶颈组织，不按论文时间排序。每篇子文章保留论文图表和实验条件。

---

## 1. 权重不能完整放进 DRAM

这条路线把模型权重从“全部常驻内存”改为“按 token 激活模式加载”。核心问题是：从 `flash/UFS` 读取权重很慢，随机小块读取更慢。方法必须同时处理预测、缓存、块化读取和计算重叠。

### 1.1 LLM in a Flash

[LLM in a Flash](/p/llm-in-flash-ssd/) 关注 `DRAM` 小于模型容量的场景。

* **核心问题：** `flash` 容量大，但到 `DRAM` 的带宽约 `1 GB/s`；`DRAM` 到 `CPU/GPU` 缓存和寄存器约 `100 GB/s`。权重搬运进入 decode 关键路径。

* **核心方法：** 方法预测 FFN 激活神经元，将需要的权重从 `flash` 读入 `DRAM`，再用 `windowing` 和 `row-column bundling` 把离散读取改成更连续的块读取。

* **实验判断：** 当可用内存约为模型容量一半时，论文报告 `CPU`、`Metal`、`NVIDIA GPU` 后端最高分别达到 `4x`、`7x`、`20x` 加速。收益主要来自减少 `Load From Flash` 时间。

* **局限：** predictor 命中率、chunk size 和存储介质绑定。模型激活模式变化后需要重新调参数。

### 1.2 PowerInfer-2

[PowerInfer-2](/p/powerinfer2-smartphone/) 关注智能手机上的大模型稀疏推理。

* **核心问题：** 逐神经元稀疏不适合移动 `NPU`，逐神经元从 `UFS` 读取会制造随机 I/O。只讨论稀疏率不能解释 decode 速度。

* **核心方法：** 方法用 `neuron cluster` 作为调度粒度。hot cluster 放入 `DRAM` 并优先交给 `NPU`，cold cluster 通过 `CPU + flash` 路径处理，I/O 与计算流水化。

* **实验判断：** 论文在 OnePlus 12 上报告 `47B` 模型达到 `11.68 tokens/s`，并报告相对已有框架最高 `27.8x` 加速。

* **局限：** cluster 计划依赖 `NPU`、`UFS`、`DRAM` 和模型激活分布。换设备要重新 profiling。

---

## 2. 移动 NPU 不能直接跑完整 Transformer

移动 `NPU` 的整数吞吐很高，但 LLM 推理还有静态图、动态 shape、attention 精度和异常值问题。可行路线通常把 `NPU` 放在稳定、低精度、可编译的子路径上，再把敏感路径留给 `CPU/GPU` 或浮点计算。

### 2.1 llm.npu

[Fast On-device LLM Inference with NPUs](asplos25-npu-llm/index.md) 研究如何把 prefill 主路径放到移动 `NPU`。

* **核心问题：** 论文测到移动 `CPU` 上 prefill 占端到端延迟 `88.3%-98.8%`，移动 `GPU` 上为 `54.2%-91.7%`。长 prompt 下不能只优化 decode。

* **核心方法：** `chunk-sharing graph` 复用静态 shape 子图；`shadow outlier execution` 把少量异常值通道留到 `CPU/GPU` 浮点路径；乱序子图调度缩短 `NPU` 关键路径。

* **实验判断：** prefill speed 图在 Redmi K70 Pro 和 Redmi K60 Pro 上比较 prompt length `64/256/1024`。论文总实验还包含 Xiaomi 14；整体报告相对 `CPU` prefill 加速 `7.3x-18.4x`，相对 `GPU` 加速 `1.3x-43.6x`。

* **局限：** 依赖 QNN 静态图、量化 kernel 和设备侧调度能力。attention 的高精度路径仍需要单独处理。

### 2.2 ShadowNPU

[ShadowNPU](/p/dynamic-sparse-attention/) 关注 `NPU-centric` 设计里的 attention。

* **核心问题：** 直接把 attention 放到低精度 `NPU` 路径会带来明显精度损失；全量回退到 `CPU/GPU` 又破坏 `NPU-centric` 的吞吐。

* **核心方法：** `NPU` 先用低成本 pilot `QK` 估计重要位置，输出 `Top-k` index；`CPU/GPU` 只对保留的 `K/V` 做稀疏 attention。

* **实验判断：** 论文报告重要位置 recall 在多个模型和稀疏率下约为 `99%+`；消融显示 pipeline、graph bucket 和 head-wise ratio 都影响最终 latency/accuracy。

* **局限：** 偏斜 attention score 是前提。长上下文或异常 head 会降低 `Top-k` 选择稳定性。

### 2.3 MobileQuant

[MobileQuant](mobilequant/index.md) 研究移动端可部署的 weight-activation quantization。

* **核心问题：** weight-only 量化仍需要浮点计算和 on-the-fly dequantization；动态 per-token activation quantization 缺少移动 `NPU/DSP` 支持。

* **核心方法：** 方法在端侧可支持的静态 per-tensor/per-channel 量化约束下，联合学习 weight equalization、weight clipping 和 activation range。线性算子尽量走 `W8A8/W4A8`，非线性边界保留 `16-bit` activation。

* **实验判断：** Galaxy S24 / Snapdragon 8 Gen 3 HTP 上，TinyLLaMA-1.1B prompt encoding latency 从 `510ms` 降到 `276ms`，energy 从 `1000mJ` 降到 `490mJ`；decode `2048` 长度 latency 从 `119ms` 降到 `95ms`。

* **局限：** 真实设备实验只覆盖 TinyLLaMA-1.1B 和一个移动 SoC。更大模型和不同 NPU compiler 需要重新验证。

---

## 3. GPU/NPU/CPU 切分需要写清数据边界

端侧 SoC 共享内存降低了复制成本，但不会消除同步和 shape 约束。异构系统的关键问题是：哪个算子在哪个设备执行，哪些中间结果要等待，切分粒度是否抵消收益。

### 3.1 HeteroLLM

[HeteroLLM](/p/heterollm-soc/) 研究移动 SoC 上 `GPU + NPU` 的 layer/tensor 级切分。

* **核心问题：** `prefill` 更像大矩阵吞吐问题，decode 更像 memory-bound `GEMV` 问题。固定后端不能同时适配两类阶段。

* **核心方法：** `HeteroInfer` 用 profiler/solver 根据 tensor shape、kernel 性能和同步成本选择 layer-level 或 tensor-level 切分，并通过 fast synchronization 缩短 GPU-NPU 等待。

* **实验判断：** 论文在 Galaxy S24 Ultra 上报告 `InternLM-1.8B` prefill 达到 `1092 token/s`；decode-heavy 的 `Llama-7B` 任务中，Hetero-tensor 相对 MNN 与 `llm.npu` 分别有 `2.06x` 和 `3.40x` 加速。

* **局限：** 性能模型与 SoC 绑定。切得过细增加同步，切得过粗损失后端特化收益。

### 3.2 Agent.xpu

[Agent.xpu](/p/agent-xpu-scheduling/) 把问题从单次 prompt latency 扩展到 agent flow 调度。

* **核心问题：** `reactive flow` 要低延迟，`proactive flow` 要后台吞吐。两类 flow 共享 `NPU`、iGPU 和 `DDR`，后台任务不能持续占住前台关键路径。

* **核心方法：** 离线生成 heterogeneous execution graph，在线由 XPU coordinator 管理队列、抢占、iGPU 仲裁和 elastic kernel 绑定。

* **实验判断：** 论文报告 proactive throughput 提升 `1.2-4.9x`，reactive latency 至少降低 `91%`。

* **局限：** 依赖 SoC 的抢占能力、静态/动态 kernel 编译方式和 `DDR` 争用模型。

---

## 4. 端侧部署要看持续负载

端侧 benchmark 如果只报一次峰值 `token/s`，会漏掉温度、功耗和频率调节。对于本地 agent，连续运行的稳定吞吐比 cold run 更接近真实部署。

### 4.1 LLM Inference at the Edge

[LLM Inference at the Edge](/p/llm-inference-edge/) 的重点是评测协议。

* **核心问题：** 手机和边缘设备的长期吞吐受 thermal management 影响。标称 TOPS 不能推出可持续 `token/s`。

* **核心方法：** 论文先 reboot、恒温、warm-up，再等 60 秒内温度变化小于 `2°C` 后开始 20 次正式测量。cold-start prefill 单独记录。

* **实验判断：** `RTX 4050` 在 runs 2-20 平均 `131.70 tok/s`，CV 为 `2.2%`；`Galaxy S24 Ultra` 平均 `10.83 tok/s`，CV 为 `6.9%`，20 次迭代消耗约 `7%` 电量。

* **局限：** 只覆盖一个 `Qwen 2.5 1.5B` 4-bit 模型和固定 prompt，不能直接外推所有端侧服务。

---

## 5. Agent 的 test-time compute 是另一类预算问题

这条路线不直接解决 `NPU` 或 `flash`，但和端侧 agent 部署有关。agent rollout 包含工具调用、代码执行、中间反思和验证。增加计算不只等于扩大 batch。

### 5.1 Scaling Test-time Compute for LLM Agents

[Scaling Test-time Compute for LLM Agents](/p/scaling-test-time/) 研究 agent 轨迹上的额外计算如何分配。

* **核心问题：** 早期 rollout 分叉会改变后续工具调用和执行成本。`best-of-N` 只覆盖一部分 test-time scaling。

* **核心方法：** 论文把计算扩展分成 parallel sampling、sequential revision、verifier/result merging 和 diversified rollouts 四类。

* **实验判断：** `GPT-4.1` baseline 为 `55.76`，`BoN` 在 search size 4 时达到 `63.03`；结果合并里 list-wise verifier 高于 voting 和独立 scoring。

* **局限：** 收益依赖 verifier 和 benchmark。轨迹越长，额外采样、反思和验证的成本越难和端侧延迟预算对齐。

---

## 6. 小结

端侧推理优化可以按资源边界分层看：

* 权重放不下：先看 `flash/UFS -> DRAM` 的读取形态，再看稀疏预测。

* `NPU` 跑不满：先看静态图、动态 shape、异常值和 attention 精度。

* 多设备协同：必须写清算子、张量、中间状态和同步位置。

* 长时间运行：用持续吞吐、温度和功耗判断部署可用性。

* agent 场景：把预算从单次 decode 扩展到 flow、rollout 和 verifier。

## 参考文献

1. Keivan Alizadeh et al., "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory", arXiv 2023. https://arxiv.org/abs/2312.11514
2. Zhenliang Xue et al., "PowerInfer-2: Fast Large Language Model Inference on a Smartphone", arXiv 2024. https://arxiv.org/abs/2406.06282
3. Daliang Xu et al., "Fast On-device LLM Inference with NPUs", ASPLOS 2025. https://arxiv.org/abs/2407.05858
4. Wangsong Yin et al., "ShadowNPU: System and Algorithm Co-design for NPU-Centric On-Device LLM Inference", arXiv 2025. https://arxiv.org/abs/2508.16703
5. Fuwen Tan et al., "MobileQuant: Mobile-friendly Quantization for On-device Language Models", Findings of EMNLP 2024. https://arxiv.org/abs/2408.13933
6. Le Chen et al., "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference", arXiv 2025. https://arxiv.org/abs/2501.14794
7. Xinming Wei et al., "Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC", arXiv 2025. https://arxiv.org/abs/2506.24045
8. Pranay Tummalapalli et al., "LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load", arXiv 2026. https://arxiv.org/abs/2603.23640
9. OPPO AI Agent Team, "Scaling Test-time Compute for LLM Agents", arXiv 2025. https://arxiv.org/abs/2506.12928
