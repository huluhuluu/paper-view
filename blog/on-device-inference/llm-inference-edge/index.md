---
title: "LLM Inference at the Edge: Mobile, NPU, and GPU Benchmarks"
date: 2026-04-02T10:33:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "llm-inference-edge"
tags: ["on-device", "Benchmarks"]
categories: ["paper"]
math: true
---

# LLM Inference at the Edge

- **LLM Inference at the Edge** 正文引言把切入点放在一个经常被忽略的问题上：always-on personal agent 真正关心的不是一次短跑式 benchmark 的峰值速度，而是连续推理几十轮之后平台还能剩下多少稳定吞吐。论文因此把实验重点压到 sustained workload characterization，而不是单次 latency，对 Qwen 2.5 1.5B 的 4-bit 版本在 Raspberry Pi 5 + Hailo-10H、Galaxy S24 Ultra、iPhone 16 Pro 与 RTX 4050 四个平台上做统一 warm-condition 评测。正文结果很清楚地揭示了不同平台的主约束：手机平台首先被 thermal management 卡住，iPhone 在极少轮次后吞吐就明显衰减，S24 Ultra 甚至会触发 OS 强制 GPU 降频；RTX 4050 的瓶颈主要是电池供电下的 power ceiling；Hailo-10H 则表现出接近 memory-bandwidth-bound 的稳定特征。实验里 RTX 4050 能稳定在约 `131.7 tok/s`、`34.1W`，Hailo-10H 则以不到 `2W` 的系统功耗维持约 `6.9 tok/s` 且方差极低，说明端侧部署的比较标准应该回到 sustained throughput、power 和 thermal behavior，而不是单独看标称 TOPS。
