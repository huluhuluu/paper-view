# Paper-view Sub-blog

端侧 AI 系统优化、LLM 推理加速、推测解码（Speculative Decoding）等系列论文阅读笔记。

## 项目结构

- `blog/`: 存放各篇论文的详细阅读笔记，按照类别（Speculative-Decoding, On-device-Inference）进行子文件夹分类。
- `index.md`: Hugo 站点的主索引页，提供专题导航。

## 专题导航

### 推测解码 (Speculative Decoding)

**完整综述**: `blog/speculative-decoding/index.md` - 从 MEDUSA 到 DFlash，系统性梳理推测解码技术演进

| 论文 | 核心贡献 |
|------|----------|
| MEDUSA | 多头并行预测，无需独立草稿模型 |
| EAGLE | 特征层自回归，重塑不确定性 |
| EAGLE-2 | 动态草稿树，上下文自适应 |
| EAGLE-3 | Training-Time Test，Scaling Law |
| DFlash | 扩散模型并行块生成 |
| P-EAGLE | 并行草稿，可扩展训练 |
| SSD | 双重推测，异步流水线 |
| SpecVocab | 词表推测，上下文感知 |
| SVIP | 自验证动态长度策略 |

### 端侧推理优化 (On-device Inference)

| 论文 | 核心贡献 |
|------|----------|
| [ASPLOS'25] Fast On-device LLM Inference with NPUs | NPU 指令级融合 |
| shadowAttn: Dynamic Sparse Attention | NPU-centric 稀疏注意力 |
| Agent.xpu | Agentic LLM 流调度 |
| HeteroLLM | 层级/张量级异构执行 |
| LLM in a Flash | 闪存为缓存 |
| LLM Inference at the Edge | NPU vs GPU Benchmark |
| Scaling Test-time Compute | Test-time 扩展 |
| PowerInfer-2 | 稀疏性利用，动态权重加载 |

## 使用说明

将此仓库作为 Hugo 站点的子模块使用，挂载到 `content/post/paper-view/`。