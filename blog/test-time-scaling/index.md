---
title: "LLM Test-Time Compute Scaling 深度解析"
date: 2026-04-02T17:00:00+08:00
lastmod: 2026-04-15T21:00:00+08:00
draft: true
slug: "test-time-scaling"
tags: ["test-time-compute", "inference-scaling", "LLM"]
categories: ["paper"]
math: true
---

# LLM Test-Time Compute Scaling 深度解析

这两篇论文关心的是同一个问题：在固定推理 FLOPs 下，额外的 test-time compute 到底应该投给更大的模型，还是投给更强的搜索与重采样过程。

- **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters** 正文引言把问题设成一个很具体的比较：如果允许模型在测试时使用一笔固定但不小的额外算力，它究竟能把困难问题提升到什么程度，以及这笔算力是否可能比继续扩参数更划算。论文的关键分析是，不同 test-time 方法的收益强烈依赖于 prompt difficulty：对较容易但还没完全答对的问题，revision/self-refinement 往往更有效，因为模型已经接近正确路径；对更难的问题，重新探索不同高层解法或对过程奖励模型做搜索更重要。因此作者把 compute-optimal policy 写成一个按题目难度自适应选择策略的过程，分别比较“修改 proposal distribution”的 revisions 路线和“优化 verifier”的 PRM search 路线，并通过 difficulty bin 来决定预算该如何分配。正文实验表明，这种按难度分配的 compute-optimal 策略相较固定 `best-of-N` 最多可提升 `4x` 的 test-time compute 利用效率，在 FLOPs 对齐实验中，小模型配合额外 test-time compute 还能在部分题目上超过约 `14x` 更大的模型。

- **Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models** 正文引言则进一步把问题系统化成“推理阶段的 scaling law”：给定一笔 inference FLOPs，模型尺寸、采样数量和搜索算法之间怎样搭配才是最优。论文先从理论上分析 majority voting 与 weighted voting 的极限行为，指出当采样数持续增加时，简单采样法最终会收敛到由模型分布本身决定的上界，这意味着只靠重复采样一定会遇到收益饱和。基于这个观察，正文系统比较了 greedy、best-of-`n`、weighted voting、MCTS 等方法，并进一步提出 `REBASE`，用 process reward model 的节点奖励来平衡树搜索扩展，不再像传统 MCTS 那样为估值付出过多 rollout 代价。实验结果显示，较小模型在低到中等预算下往往更 compute-optimal，而随着预算增长，大模型才开始接管优势；更重要的是，`REBASE` 在 MATH 与 GSM8K 上持续形成更优的 cost-performance 曲线，例如 Llemma-7B 配合 `REBASE` 可以在多个预算区间里稳定压过 Llemma-34B 配合常规采样策略。
