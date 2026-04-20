---
title: "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory"
date: 2026-04-02T10:22:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "llm-in-flash-ssd"
tags: ["on-device", "Memory-Limited"]
categories: ["paper"]
math: true
---

# LLM in a Flash

- **LLM in a Flash** 正文引言首先把问题定在“模型参数比设备 DRAM 更大”这条硬边界上：像 7B 级模型仅半精度权重就超过十几 GB，单靠量化并不能改变“整模型需要同时驻留 DRAM”这一前提。论文随后专门分析了 flash 与 DRAM 的差异，指出 flash 的顺序读带宽虽然不低，但随机小块读取的首字节延迟和吞吐衰减会迅速放大，因此如果仍按矩阵的常规访问方式逐块搬运权重，端侧推理会被 I/O 完全拖住。围绕这一点，方法部分先建立同时考虑 chunk size 和传输量的 inference cost model，再提出三类优化：只把 attention 和 embedding 常驻 DRAM，对 FFN 依赖低秩 predictor 做按需加载；用 `windowing` 复用最近若干 token 对应的激活神经元，减少重复搬运；再用 `row-column bundling` 和专门的数据结构把 flash 访问改写成更大的顺序块。正文实验显示，这套设计能够在只提供约一半模型容量 DRAM 的情况下运行约 `2x` 于可用内存的模型，并相对 naive loading 在 CPU 和 GPU 后端分别获得最高约 `4x` 和 `20x` 的推理提速。
