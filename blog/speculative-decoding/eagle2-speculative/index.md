---
title: "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"
date: 2026-04-02T10:12:00+08:00
slug: "eagle2-speculative"
tags: ["speculative-decoding"]
categories: ["paper"]
math: true
---

## 1. 核心贡献 (Core Contributions)

EAGLE-2 进一步完善了 EAGLE 框架，其核心贡献在于提出了 **动态草稿树 (Dynamic Draft Trees)**。解决了推测解码在处理逻辑复杂或歧义较多的文本时效果不佳的问题。

## 2. 核心方法 (Methods)

### 2.1 动态树结构
不同于 EAGLE-1 或 Medusa 使用固定的树形状（Tree Topology），EAGLE-2 会在每一步生成时评估当前 token 的置信度（Confidence Score）。
- **确定性场景**: 树会变得很深，进行长路径推测。
- **不确定场景**: 树会变宽，探索更多可能的分支。

### 2.2 置信度评估
利用预测分布的 Entropy 或 Top-1 概率来动态分配算力。

## 3. 深度分析与对比 (Analysis & Comparison)

### 分析优势：
EAGLE-2 的最大优势在于其**自适应性**。在处理代码或专业文献等确定性强的任务时，其加速比可达 3x 以上；而在闲聊等发散性任务中，它通过变宽的树结构有效维持了接受率，避免了频繁的回退。

这种动态性标志着推测解码从“暴力推测”向“启发式搜索”的转变。相比于云侧常用的并行大 Batch，EAGLE-2 在端侧（Batch=1）场景下的优势尤为突出。
