# Paper-view

端侧 AI 系统优化、LLM 推理加速、推测解码（Speculative Decoding）等系列论文阅读笔记。

---

## 1. 推测解码 (Speculative Decoding)

推测解码是"空间换时间"的并行化加速路径，使用轻量级草稿模型快速生成候选 token，然后由目标模型并行验证。

| 论文 | 核心贡献 | 链接 |
|------|----------|------|
| MEDUSA | 多头并行预测，无需独立草稿模型 | [传送门](blog/speculative-decoding/index.md) |
| EAGLE | 特征层自回归，重塑不确定性 | [传送门](blog/speculative-decoding/index.md) |
| EAGLE-2 | 动态草稿树，上下文自适应 | [传送门](blog/speculative-decoding/index.md) |
| EAGLE-3 | Training-Time Test，Scaling Law | [传送门](blog/speculative-decoding/index.md) |
| DFlash | 扩散模型并行块生成 | [传送门](blog/speculative-decoding/index.md) |
| P-EAGLE | 并行草稿，可扩展训练 | [传送门](blog/speculative-decoding/index.md) |
| SSD | 双重推测，异步流水线 | [传送门](blog/speculative-decoding/index.md) |
| SpecVocab | 词表推测，上下文感知 | [传送门](blog/speculative-decoding/index.md) |
| SVIP | 自验证动态长度策略 | [传送门](blog/speculative-decoding/index.md) |

---

## 2. 端侧推理优化 (On-device Inference)

聚焦于智能手机及边缘计算设备（Mobile SoCs）上的极速推理实践。

| 论文 | 核心贡献 | 链接 |
|------|----------|------|
| Fast On-device LLM Inference with NPUs | NPU 指令级融合，20+ tokens/s | [传送门](blog/on-device-inference/asplos25-npu-llm/index.md) |
| shadowAttn: Dynamic Sparse Attention | NPU-centric 稀疏注意力，6.9× 加速 | [传送门](blog/on-device-inference/dynamic-sparse-attention/index.md) |
| Agent.xpu | Agentic LLM 流调度，异构协同 | [传送门](blog/on-device-inference/agent-xpu-scheduling/index.md) |
| HeteroLLM | 层级/张量级异构执行，9.99× 加速 | [传送门](blog/on-device-inference/heterollm-soc/index.md) |
| LLM in a Flash | 闪存为缓存，突破内存限制 | [传送门](blog/on-device-inference/llm-in-flash-ssd/index.md) |
| LLM Inference at the Edge | NPU vs GPU Benchmark | [传送门](blog/on-device-inference/llm-inference-edge/index.md) |
| Scaling Test-time Compute | Test-time 扩展，小模型大能力 | [传送门](blog/on-device-inference/scaling-test-time/index.md) |
| PowerInfer-2 | 稀疏性利用，动态权重加载 | [传送门](blog/on-device-inference/powerinfer2-smartphone/index.md) |

---

## 使用说明

将此仓库作为 Hugo 站点的子模块使用，挂载到 `content/post/paper-view/`。
