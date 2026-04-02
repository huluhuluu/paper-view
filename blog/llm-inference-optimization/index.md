---
title: "LLM 推理系统优化深度解析"
date: 2026-04-02T17:10:00+08:00
slug: "llm-inference-optimization"
tags: ["inference", "FlashAttention", "PagedAttention", "vLLM"]
categories: ["paper"]
math: true
---

# LLM 推理系统优化深度解析

LLM 推理优化是连接模型与实际应用的关键环节。本文将从理论基础（Roofline 模型）出发，深入解析 FlashAttention、PagedAttention/vLLM 和 Continuous Batching 等核心技术的原理与实现。

---

## 一、理论基础：Roofline 模型

### 1.1 模型概述

Roofline 模型是一个可视化性能模型，用于分析程序在计算与内存访问之间的瓶颈。

**核心指标**：

**算术强度（Arithmetic Intensity）**：
$$I = \frac{\text{FLOPs}}{\text{Bytes}}$$

即每字节内存传输对应的浮点运算次数。

**Roofline 公式**：
$$P \leq \min(\pi, I \times \beta)$$

其中：
- $P$：可达性能（FLOPS）
- $\pi$：峰值计算能力
- $\beta$：内存带宽
- $I$：算术强度

### 1.2 两种瓶颈区域

| 区域 | 条件 | 特点 | 优化方向 |
|------|------|------|----------|
| **Memory-Bound** | $I < \pi/\beta$ | 性能受限于内存带宽 | 减少数据移动 |
| **Compute-Bound** | $I \geq \pi/\beta$ | 性能受限于计算能力 | 并行化、向量化 |

### 1.3 LLM 推理的 Roofline 分析

**Prefill 阶段**（处理输入 prompt）：
- 高算术强度（序列并行计算）
- 通常 Compute-Bound

**Decode 阶段**（逐 token 生成）：
- 低算术强度（每个 token 需要加载全部权重）
- 严重 Memory-Bound

**关键洞察**：Decode 阶段的主要瓶颈是**内存带宽**，而非计算能力。这指导了后续所有优化方向。

---

## 二、FlashAttention: IO 感知的注意力计算

### 2.1 核心问题

标准注意力计算：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

内存复杂度：$O(N^2)$，其中 $N$ 是序列长度。

**IO 瓶颈**：
- 需要在 HBM（高带宽内存）和 SRAM（片上缓存）之间多次读写
- 中间结果 $QK^T$ 和 softmax 结果占用大量 HBM

### 2.2 核心创新：分块计算（Tiling）

**关键思想**：将注意力计算分块，在 SRAM 中完成核心计算，减少 HBM 访问。

**算法流程**：

```
For each block of Q:
    For each block of K, V:
        1. 加载 Q_block, K_block, V_block 到 SRAM
        2. 在 SRAM 中计算局部注意力
        3. 在线 softmax 更新
        4. 累积结果
    写回最终结果到 HBM
```

### 2.3 IO 复杂度分析

| 方法 | HBM 访问次数 | 内存占用 |
|------|-------------|----------|
| 标准注意力 | $O(N^2)$ | $O(N^2)$ |
| FlashAttention | $O(N^2 d / M)$ | $O(N)$ |

其中 $M$ 是 SRAM 大小，$d$ 是隐藏维度。

**关键**：虽然计算量相同，但 HBM 访问大幅减少，实现 2-4× 加速。

### 2.4 实验结果

| 模型 | 任务 | 加速比 | 内存节省 |
|------|------|--------|----------|
| BERT-large | 训练 | 15% | - |
| GPT-2 | 训练 (seq=1K) | 3× | 5-20× |
| 长序列 | seq=16K-64K | 2.4× | 显著 |

**新能力解锁**：
- Path-X 挑战（seq=16K）：首次超越随机猜测，61.4% 准确率
- Path-256（seq=64K）：63.1% 准确率

---

## 三、PagedAttention / vLLM: 高效内存管理

### 3.1 核心问题：KV Cache 的内存浪费

**传统 KV Cache 管理的问题**：

1. **预分配浪费**：为最大序列长度预分配，实际往往更短
2. **内存碎片**：连续分配导致碎片化
3. **无法共享**：相同前缀的请求无法共享 KV Cache

**量化影响**：
- 内存利用率低（通常 < 50%）
- 限制了最大 batch size
- 吞吐量受限

### 3.2 核心创新：分页式内存管理

受操作系统虚拟内存启发，PagedAttention 将 KV Cache 组织为**固定大小的块（Block）**：

```
逻辑 KV Cache: [Block 0] → [Block 1] → [Block 2] → ...
                     ↓            ↓            ↓
物理内存:      [Page 5]     [Page 3]     [Page 7]
```

**关键特性**：

1. **按需分配**：仅在实际需要时分配块
2. **非连续存储**：逻辑连续，物理可分散
3. **引用计数**：支持块共享和写时复制

### 3.3 共享 KV Cache

**场景**：多轮对话或 beam search

传统方法：每个候选独立存储完整 KV Cache

PagedAttention：
- 共享公共前缀的 KV Cache 块
- 仅在分支点复制
- 内存节省可达 55%

### 3.4 vLLM 系统架构

```
┌─────────────────────────────────────┐
│           Scheduler                  │
│  (iteration-level scheduling)        │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│         Block Manager                │
│  (KV cache allocation & sharing)     │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│       PagedAttention Kernel          │
│  (block-wise attention compute)      │
└─────────────────────────────────────┘
```

### 3.5 实验结果

| 基准 | 吞吐提升 | 延迟 |
|------|----------|------|
| vs FasterTransformer | 2-4× | 相当 |
| vs HuggingFace | 10-20× | 相当 |

**内存效率**：接近零浪费的 KV Cache 内存使用。

---

## 四、Continuous Batching: 迭代级调度

### 4.1 核心问题：静态批处理的低效

**传统静态批处理**：

```
Batch 开始: [Request 1, Request 2, Request 3, Request 4]
                ↓
Request 2 完成，但必须等待其他请求
                ↓
Batch 结束: 所有请求完成后才能开始新 batch
```

**问题**：
- 早完成的请求占用资源
- 新请求必须等待当前 batch 完成
- GPU 利用率波动大

### 4.2 核心创新：迭代级调度（Iteration-Level Scheduling）

**关键思想**：在每次迭代（生成一个 token）后重新调度，而非等待整个请求完成。

```
Iteration 1: [Request 1, Request 2, Request 3, Request 4]
Iteration 2: [Request 1, Request 2, Request 3, Request 4]
Iteration 3: [Request 1, Request 3, Request 4] + [Request 5]
                    ↑ Request 2 完成，Request 5 加入
```

### 4.3 Selective Batching

并非所有操作都适合批处理：
- Attention 操作：序列长度不同，难以批处理
- Linear 层：可轻松批处理

**解决方案**：
- Attention：单独计算或分组计算
- Linear：合并为大批量矩阵乘法

### 4.4 ORCA 系统架构

```
┌─────────────────────────────────────┐
│         Iteration Scheduler          │
│  (schedule per iteration)            │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│         Selection Engine             │
│  (select ops for batching)           │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│       Execution Engine               │
│  (run model iteration)               │
└─────────────────────────────────────┘
```

### 4.5 实验结果

| 基准 | 吞吐提升 | 场景 |
|------|----------|------|
| vs FasterTransformer | 36.9× | GPT-3 175B |
| vs 静态批处理 | 5-10× | 变长输出 |

---

## 五、技术协同与系统设计

### 5.1 优化层次

| 层次 | 技术 | 目标 |
|------|------|------|
| **Kernel 层** | FlashAttention | 减少内存访问 |
| **内存层** | PagedAttention | 高效内存管理 |
| **调度层** | Continuous Batching | 提高资源利用 |

### 5.2 组合效果

| 组合 | 效果 |
|------|------|
| FlashAttention + PagedAttention | 内存效率 + 计算效率 |
| PagedAttention + Continuous Batching | vLLM 的核心 |
| 全部组合 | 最先进的 LLM 服务系统 |

### 5.3 现代系统架构

```
┌─────────────────────────────────────────────┐
│                  API Layer                   │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│              Scheduler                       │
│  • Continuous Batching                       │
│  • Priority-based Scheduling                 │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│           Memory Manager                     │
│  • PagedAttention                            │
│  • KV Cache Sharing                          │
└────────────────────┬────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│            Kernel Layer                      │
│  • FlashAttention                            │
│  • Optimized GEMM                            │
└─────────────────────────────────────────────┘
```

---

## 六、未来方向

### 6.1 硬件协同设计

1. **HBM 容量增长**：缓解内存瓶颈
2. **片上缓存扩大**：支持更大分块
3. **专用加速器**：针对注意力计算的硬件

### 6.2 算法创新

1. **稀疏注意力**：减少计算和内存访问
2. **推测解码**：利用小模型加速大模型推理
3. **分层缓存**：CPU/GPU/SSD 多级存储

### 6.3 系统优化

1. **多模型服务**：资源共享与隔离
2. **弹性伸缩**：动态调整资源
3. **联邦推理**：跨设备协同

---

## 七、总结

LLM 推理优化是一个多层次系统工程：

| 层次 | 核心洞察 | 代表技术 |
|------|----------|----------|
| **理论** | Decode 阶段 Memory-Bound | Roofline 分析 |
| **Kernel** | 减少 HBM 访问 | FlashAttention |
| **内存** | 按需分配、共享 | PagedAttention |
| **调度** | 迭代级调度 | Continuous Batching |

这些技术共同推动了 LLM 服务从"能用"到"高效"的跨越，使大规模模型服务成为可能。

---

## 参考文献

1. Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 2009
2. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022
3. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
4. Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022
