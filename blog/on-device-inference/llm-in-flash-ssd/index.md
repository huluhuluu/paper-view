---
title: "LLM in a flash: Efficient Large Language Model Inference with Limited Memory"
date: 2026-04-02T10:22:00+08:00
slug: "llm-in-flash-ssd"
tags: ["on-device", "Memory-Limited"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

Apple 团队出品，解决了 4GB/8GB 内存手机运行超大模型的问题。其核心贡献：
1. **闪存为缓存 (Flash as Cache)**: 将模型参数存放在闪存中。
2. **窗口式加载 (Windowing)**: 只将当前计算层附近的权重常驻内存，随推理进程滑动更新。
3. **列式捆绑 (Row-Column Bundling)**: 优化 SSD 读取模式，最大化闪存的顺序读吞吐量。

## 2. 深度分析 (Analysis)

### 为什么这篇文章具有开创性？
它彻底否定了“模型必须全装进内存”的传统信条。通过精密的访存流水线设计，它让“内存”成为了权重的“高速中转站”。
- **优势**: 极大降低了手机硬件门槛。
- **对比**: 相比 PowerInfer-2，本篇更偏向于存储层级的底层管理。

这种“空间换内存”的思想，对于目前内存配置仍然较紧俏的移动端生态系统具有深远的工程指导意义。
