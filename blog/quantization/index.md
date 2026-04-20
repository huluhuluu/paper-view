---
title: "LLM 量化技术深度解析"
date: 2026-04-02T16:00:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "llm-quantization"
tags: ["quantization", "LLM", "inference-optimization"]
categories: ["paper"]
math: true
---

# LLM 量化技术深度解析

这三篇论文处理的是同一类问题，但切入点并不一样：`LLM.int8()` 先解释多亿参数模型里为什么常规 8-bit 会失效，`SmoothQuant` 进一步把 activation 量化困难写成可转移的尺度问题，`AWQ` 则把 weight-only 量化的关键从“所有权重”收缩到“少量真正重要的通道”。

- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** 正文引言先把压力集中到 Transformer 推理最重的前馈层和 attention projection 上，并指出多亿参数区间里无损 8-bit 量化仍然是开放问题。论文随后给出的关键分析并不是普通量化噪声，而是 hidden state 中极少数但高度系统化的 outlier feature：它们在 `6.7B` 左右开始扩散到全部层，幅值可达普通维度的 `20x`，而且集中在不超过 6 到 7 个 hidden dimensions；正文还通过把这些维度置零的实验说明，它们会显著拉低 attention 的 top-1 概率并带来 `600%-1000%` 的 perplexity 恶化。基于这个观察，方法部分把 `LLM.int8()` 分成两段：一段是按 row/column 做归一化的 vector-wise quantization，另一段是把阈值以上的 outlier 维度拆到 16-bit 路径单独计算，保留其余 `99.9%` 数值在 8-bit 上完成乘法。实验部分显示，这套设计是少数能把 OPT 从 `125M` 一路扩到 `175B` 仍维持接近 16-bit 准确率的方法，同时把模型显存压缩到约原来的 `1/2`。

- **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** 正文引言把问题界定得很清楚：LLM 推理需要的不只是更小的权重存储，还需要让所有核心矩阵乘法真正落到高效的 `W8A8` kernel 上，但现有 PTQ 一旦碰到大模型 activation 就会失稳。论文在分析部分进一步指出，困难并不对称地来自 activation：outlier 会把整列量化范围拉宽，使大量正常值只剩下极少的有效量化级别；而权重分布相对平坦，因此只做 per-token activation quantization 并不能解决问题，真正有用的是 per-channel 级别的平滑，但这又与 INT8 GEMM 的实现方式不兼容。`SmoothQuant` 的核心就是把这个矛盾写成一个离线等价变换，用 `s_j = max(|X_j|)^α / max(|W_j|)^{1-α}` 这样的 per-channel scale 把量化难度从 activation 迁移到 weight，并把这一步提前融合到前一层参数里，从而在运行时保留原始线性映射却让 activation 变得更平滑。正文实验表明，该方法在 OPT-175B、BLOOM-176B、GLM-130B 以及后续 Llama、Falcon、Mistral 等模型上都能保持接近原模型的零样本性能，并在 FasterTransformer 上取得最高 `1.56x` 加速与近 `2x` 显存节省。

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** 正文引言把场景直接放到端侧部署：如果模型参数远超边缘设备内存，真正可落地的是 low-bit weight-only quantization，但这一路线的核心难点在于低比特下精度损失很容易集中爆发。论文前几页的关键分析不是从权重分布出发，而是从 activation 统计出发：作者发现只有 `0.1%-1%` 的 salient weights 对量化误差最敏感，而这些权重通道应当通过 activation magnitude 来识别，而不是通过 weight norm；正文表 1 也说明，仅仅把少量由 activation 决定的重要通道保留为高精度，就足以把 INT3 量化后的 perplexity 大幅拉回。为了避免混合精度实现带来的硬件低效，`AWQ` 在方法部分进一步推导出 scaling-based 保护方案，即通过 per-channel scaling 放大 salient channel、降低其相对量化误差，再用小规模 grid search 选择每层的缩放强度，整个过程不依赖 reconstruction 或反向传播。正文实验显示，`AWQ` 在 LLaMA、Llama-2、Mistral、Mixtral 以及指令微调、多模态模型上都优于 RTN 与 GPTQ；配合 `TinyChat` 的 on-the-fly dequantization、SIMD-aware packing 和 kernel fusion，桌面与移动 GPU 上可获得 `3x` 以上的实际推理加速。
