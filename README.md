# Paper-view

论文阅读笔记。

---

## 1. 推测解码 (Speculative Decoding)

推测解码是"空间换时间"的并行化加速路径，使用轻量级草稿模型快速生成候选 token，然后由目标模型并行验证。

| 论文 | 核心贡献 | 状态 | 链接 |
|------|----------|------|------|
| MEDUSA / EAGLE / EAGLE-2 / EAGLE-3 / DFlash / P-EAGLE / SVIP | 推测解码主线与并行化路线 | 🚧 TODO | [传送门](blog/speculative-decoding/index.md) |
| Scaling Law / 小模型实证 / MAGICDEC / Hidden State SD / SSD | 草稿模型训练与加速建模 | ✅ 已完成 | [传送门](blog/speculative-decoding/draft-train/index.md) |

---

## 2. 端侧推理优化 (On-device Inference)

聚焦于智能手机及边缘计算设备（Mobile SoCs）上的极速推理实践。

| 论文 | 核心贡献 | 状态 | 链接 |
|------|----------|------|------|
| Fast On-device LLM Inference with NPUs | NPU 指令级融合，20+ tokens/s | ✅ 已完成 | [传送门](blog/on-device-inference/asplos25-npu-llm/index.md) |
| shadowAttn: Dynamic Sparse Attention | NPU-centric 稀疏注意力，6.9× 加速 | 🚧 TODO | [传送门](blog/on-device-inference/dynamic-sparse-attention/index.md) |
| Agent.xpu | Agentic LLM 流调度，异构协同 | 🚧 TODO | [传送门](blog/on-device-inference/agent-xpu-scheduling/index.md) |
| HeteroLLM | 层级/张量级异构执行，9.99× 加速 | 🚧 TODO | [传送门](blog/on-device-inference/heterollm-soc/index.md) |
| LLM in a Flash | 闪存为缓存，突破内存限制 | 🚧 TODO | [传送门](blog/on-device-inference/llm-in-flash-ssd/index.md) |
| LLM Inference at the Edge | NPU vs GPU Benchmark | 🚧 TODO | [传送门](blog/on-device-inference/llm-inference-edge/index.md) |
| Scaling Test-time Compute | Test-time 扩展，小模型大能力 | 🚧 TODO | [传送门](blog/on-device-inference/scaling-test-time/index.md) |
| PowerInfer-2 | 稀疏性利用，动态权重加载 | 🚧 TODO | [传送门](blog/on-device-inference/powerinfer2-smartphone/index.md) |

---

## 3. 其他专题

| 方向 | 核心贡献 | 状态 | 链接 |
|------|----------|------|------|
| LLM 推理系统优化 | `Roofline`、`FlashAttention`、`PagedAttention`、`Continuous Batching` | 🚧 TODO | [传送门](blog/llm-inference-optimization/index.md) |
| LLM 量化技术 | `LLM.int8()`、`SmoothQuant`、`AWQ` | 🚧 TODO | [传送门](blog/quantization/index.md) |
| LoRA 微调路线 | `LoRA`、`QLoRA`、`PISSA`、`LoRA-GA` | 🚧 TODO | [传送门](blog/lora-finetuning/index.md) |
| Test-time Scaling | `Scaling LLM Test-Time Compute Optimally` 与 `Inference Scaling Laws` | 🚧 TODO | [传送门](blog/test-time-scaling/index.md) |

---

## 使用说明

将此仓库作为 Hugo 站点的子模块使用，挂载到 `content/post/paper-view/`。
