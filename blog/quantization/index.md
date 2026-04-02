---
title: "LLM 量化技术深度解析"
date: 2026-04-02T16:00:00+08:00
slug: "llm-quantization"
tags: ["quantization", "LLM", "inference-optimization"]
categories: ["paper"]
math: true
---

# LLM 量化技术深度解析

量化（Quantization）是大语言模型部署中最重要的优化技术之一。通过将高精度浮点数转换为低精度整数，量化可以显著减少内存占用和加速推理。本文将深入解析三篇代表性论文：LLM.int8()、SmoothQuant 和 AWQ，揭示 LLM 量化的核心技术演进。

---

## 一、LLM.int8(): 8-bit 矩阵乘法的突破

### 1.1 核心问题

大语言模型的推理需要大量 GPU 内存。以 GPT-3 175B 为例，仅加载模型就需要至少 350GB 内存。传统的 8-bit 量化方法在小模型上表现良好，但在 6.7B 参数以上的模型上会出现严重的性能下降。

**关键发现**：当 Transformer 模型规模超过 6.7B 参数时，会在特征维度中出现**系统性离群值（Systematic Outliers）**。这些离群值：
- 出现在约 75% 的 Transformer 层中
- 集中在仅 6 个特征维度（占总维度的 0.1%）
- 幅度可达其他维度的 20 倍以上
- 对模型性能至关重要：将其置零会导致困惑度上升 600-1000%

### 1.2 核心方法：Vector-wise Quantization + Mixed-Precision Decomposition

LLM.int8() 提出两阶段量化策略：

**阶段一：向量级量化（Vector-wise Quantization）**

将矩阵乘法视为一系列独立的内积操作，为每个内积使用独立的归一化常数：

$$C_{f16} \approx C_{i32} = S \cdot A_{i8} B_{i8}$$

其中 $S = c_x \otimes c_w$ 是行列归一化常数的外积。这种方法在 2.7B 以下模型效果良好。

**阶段二：混合精度分解（Mixed-Precision Decomposition）**

对于 6.7B 以上的模型，将特征维度分解为离群值和正常值两部分：

$$C_{f16} \approx \sum_{h \in O} X^h_{f16} W^h_{f16} + S \cdot \sum_{h \notin O} X^h_{i8} W^h_{i8}$$

其中 $O$ 是离群值特征维度集合（通常 $|O| \leq 7$），阈值 $\alpha = 6.0$。

### 1.3 关键洞察

1. **离群值的系统性**：离群值不是随机出现的，而是高度系统化的，几乎出现在所有序列维度但仅限于特定的特征维度。

2. **重要性验证**：移除这些离群值特征会导致严重的性能下降，而移除相同数量的随机特征几乎没有影响。

3. **内存效率**：由于离群值维度极少（约 0.1%），混合精度分解仅增加约 0.1% 的内存开销。

### 1.4 实验结果

- 在 OPT-175B 上实现零性能损失的 8-bit 推理
- 内存占用减少 50%（相比 FP16）
- 使单服务器消费级 GPU 运行 OPT-175B/BLOOM 成为可能

---

## 二、SmoothQuant: 激活平滑的智慧

### 2.1 核心问题

LLM.int8() 虽然解决了精度问题，但混合精度分解在硬件加速器上难以高效实现。能否找到一种纯 INT8 方案？

**关键观察**：
1. **权重易于量化**：权重分布均匀平坦
2. **激活难以量化**：存在大幅度离群值
3. **离群值的持久性**：一旦某个通道出现离群值，它在所有 token 中持续存在
4. **通道内方差小**：同一通道内的幅度变化很小

### 2.2 核心方法：平滑迁移量化难度

SmoothQuant 的核心思想是将激活的量化难度**迁移**到权重上：

$$Y = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s) W) = \hat{X} \hat{W}$$

通过数学等价变换，平滑因子 $s$ 将激活中的离群值缩放转移给权重。

**平滑因子计算**：

$$s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}$$

其中 $\alpha$ 是迁移强度超参数，控制从激活迁移多少难度到权重：
- $\alpha = 0$：仅量化权重（激活保持 FP16）
- $\alpha = 1$：完全迁移到权重
- 实验表明 $\alpha = 0.5$ 对大多数模型效果最佳

### 2.3 为什么有效？

| 方法 | 激活量化 | 权重量化 | 硬件效率 |
|------|----------|----------|----------|
| Per-tensor | 困难（离群值主导） | 容易 | 高 |
| Per-token | 稍好 | 容易 | 高 |
| Per-channel | 容易 | 容易 | **不支持**（无法用于 GEMM） |
| SmoothQuant | 容易 | 容易 | **高** |

Per-channel 激活量化理论上最优，但无法映射到高效的 INT8 GEMM 内核。SmoothQuant 通过离线变换，在保持硬件效率的同时获得了 Per-channel 的量化效果。

### 2.4 实验结果

- 在 OPT-175B、BLOOM-176B、GLM-130B 上实现无损 W8A8 量化
- 相比 FP16 实现 1.56× 加速和 2× 内存节省
- 首次实现 530B 模型在单节点内服务
- 可与 FasterTransformer 集成

---

## 三、AWQ: 激活感知的权重量化

### 3.1 核心问题

SmoothQuant 解决了 W8A8 量化问题，但如果只需要量化权重呢？权重-only 量化在内存受限场景（如端侧部署）中非常重要。

**关键发现**：并非所有权重都同等重要。保护仅 1% 的显著权重可以大幅减少量化误差。

### 3.2 核心方法

**观察 1：激活分布指导权重重要性**

通过分析激活分布识别显著权重通道：
- 显著性取决于激活幅度，而非权重本身
- 激活幅度大的通道对量化误差更敏感

**观察 2：等效缩放保护显著通道**

避免硬件低效的混合精度，通过数学等效变换保护显著通道：

$$\text{Quant}(w \cdot s) \cdot (x / s) \approx w \cdot x$$

将显著通道的权重放大 $s$ 倍，同时将对应输入缩小 $s$ 倍，等效于增加了量化精度。

**平滑因子确定**：

$$s = \max(|X|)^\alpha / \max(|W|)^{1-\alpha}$$

离线收集激活统计信息确定每个通道的平滑因子。

### 3.3 AWQ vs LLM.int8() vs SmoothQuant

| 特性 | LLM.int8() | SmoothQuant | AWQ |
|------|------------|-------------|-----|
| **量化类型** | W8A8（混合精度） | W8A8 | W4A16 / W8A16 |
| **硬件友好** | 一般 | 好 | 很好 |
| **训练需求** | 无 | 无 | 无 |
| **适用场景** | 大模型推理 | 大模型推理 | 端侧部署 |

### 3.4 实验结果

- INT4 量化几乎无损（困惑度增加 < 1%）
- 相比 FP16 实现 3× 以上加速
- 首次实现多模态 LLM 的高精度量化
- 支持 LLaMA、OPT、BLOOM 等主流模型

### 3.5 TinyChat 框架

AWQ 配套的推理框架 TinyChat：
- 内核融合和平台感知权重打包
- 在桌面和移动 GPU 上实现 3× 以上加速
- 使 70B 模型在移动 GPU 上运行成为可能

---

## 四、技术演进总结

### 4.1 核心洞察

| 论文 | 核心洞察 |
|------|----------|
| LLM.int8() | 大模型存在系统性离群值，需特殊处理 |
| SmoothQuant | 离群值通道在各 token 间一致，可离线平滑 |
| AWQ | 1% 权重决定量化精度，可通过缩放保护 |

### 4.2 方法对比

```
LLM.int8()  →  发现离群值问题，混合精度方案
      ↓
SmoothQuant →  纯 INT8 方案，迁移量化难度
      ↓
AWQ         →  权重-only 方案，激活感知缩放
```

### 4.3 适用场景

| 场景 | 推荐方法 |
|------|----------|
| 云端大模型推理 | SmoothQuant (W8A8) |
| 端侧部署 | AWQ (W4A16) |
| 内存极度受限 | AWQ + QLoRA |

---

## 参考文献

1. Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", NeurIPS 2022
2. Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", ICML 2023
3. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", MLSys 2024
