---
title: "推测解码 系列论文深度解析"
date: 2026-04-02T12:00:00+08:00
lastmod: 2026-04-02T12:00:00+08:00
slug: "speculative-decoding-overview"
tags: ["speculative-decoding", "LLM-Inference", "LLM-Acceleration"]
categories: ["paper"]
math: true
---

# 推测解码系列论文深度解析

推测解码 是近年来大语言模型推理加速领域最重要的技术突破之一。其核心思想是"空间换时间"——使用轻量级草稿模型快速生成候选 token，然后由目标模型并行验证，从而打破自回归生成的串行瓶颈。本文将系统性地解析该领域的代表性工作，从基础框架到最新进展。

---

## 一、基础框架：MEDUSA 与多头并行预测

### 1.1 MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

**核心贡献：**

MEDUSA 针对大语言模型（LLM）自回归生成的串行瓶颈，提出了一种简洁而高效的加速框架：

1. **多头结构**: 不同于传统的 Speculative Decoding 需要维护一个独立的草稿模型（Draft Model），MEDUSA 直接在主模型的最后一个隐藏层上并行挂载了多个 MLP 头。
2. **无需草稿模型**: 极大地降低了系统复杂度，避免了模型间同步和草稿模型占用的额外显存。
3. **Tree Attention**: 提出了一种树状注意力机制，允许在一次大模型的前向传递中验证多个候选 token 路径。

**核心方法：**

在 Llama 等主模型之上，MEDUSA 挂载了 $k$ 个独立的 MLP 头。每个头负责预测未来第 $i$ 个位置（$1 \le i \le k$）的 token 分布。预测公式如下：

$$p_t^{(i)} = \text{softmax}(W_2^{(i)} \cdot \text{SiLU}(W_1^{(i)} \cdot h_t) + h_t)$$

其中 $h_t$ 是主模型第 $t$ 步的隐藏状态，$W_1^{(i)} \in \mathbb{R}^{d \times d}$，$W_2^{(i)} \in \mathbb{R}^{d \times V}$。

MEDUSA 并不简单地只预测一条路径，而是生成多个候选并构建出一棵"草稿树"。通过构建一个特殊的 $N \times N$ 的 Attention Mask，MEDUSA 可以在单次推理中并行验证树上的所有节点。

**训练策略：**

MEDUSA 采用"冻结主模型，训练头"的策略。这种方式不仅保护了主模型的生成能力，还能在极小的数据集（如 ShareGPT）上快速收敛。

**优势与局限：**

- **延迟极低**: 相比传统的 Speculative Decoding（如 Leviathan 等），MEDUSA 避免了草稿模型生成 token 时的等待时间。
- **内存友好**: 只有一个模型，KV Cache 管理更加直接。

然而，MEDUSA 的主要缺陷在于其预测是"静态"的。由于每个 Medusa Head 仅基于当前的隐藏状态 $h_t$ 进行独立预测，忽略了预测过程中 token 间的相互依赖关系。这导致当推测步数 $k$ 增大时，准确率会呈指数级下降。

---

## 二、特征层自回归：EAGLE 系列的范式转移

### 2.1 EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

**核心贡献：**

EAGLE 在推测解码领域引入了范式转移：

1. **特征层自回归**: 提出不在词表层做推测，而是在特征层进行自回归推测。
2. **重塑不确定性**: 论证了特征层的不确定性远低于词表层，从而能实现更高的接受率。

**关键洞察：**

论文提出了两个关键观察：
1. **特征层的自回归比词表层更简单**: 特征序列比 token 序列展现更强的规律性。在特征层进行自回归处理，然后使用原始 LLM 的 LM Head 推导 token，比直接预测 token 效果更好。
2. **采样过程的不确定性**: 在文本生成中，目标 LLM 预测 token 分布并采样，引入随机性。不同的采样 token（如 "am" 或 "always"）导致不同的特征序列，这给特征层自回归带来了歧义。

**结构设计：**

EAGLE 包含一个极轻量级的自回归组件（通常只有 1 层 Transformer）。其输入是当前 token 的嵌入向量 $e_t$ 和前一层的隐藏状态 $h_{t-1}$，以及一个时间步提前的 token 序列来解决不确定性问题。

**为什么 EAGLE 比 Medusa 强？**

Medusa 的每个头是并行的、相互独立的。而 EAGLE 是自回归的，能够捕获更丰富的上下文信息：
- **数据证明**: 在 Llama-2-70B 上，EAGLE 的接受率通常能达到 80% 以上，而 Medusa 在长序列预测时准确率会骤降。
- **计算开销**: 虽然 EAGLE 引入了自回归，但因为操作的是 Hidden States 而非完整的词表，其单步延迟极低。

EAGLE 实际上是在模仿大模型的"思维路径"。通过在更稳定的高维特征空间进行推测，它极大地缓解了自回归生成中的"累积误差"问题。

### 2.2 EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees

**核心贡献：**

EAGLE-2 进一步完善了 EAGLE 框架，提出了 **动态草稿树**，解决了推测解码在处理逻辑复杂或歧义较多的文本时效果不佳的问题。

**关键发现：**

论文发现一个重要现象：**草稿 token 的接受率不仅与位置相关，还高度依赖于上下文**。

传统的静态草稿树（如 Sequoia、EAGLE-1、Medusa）假设接受率只与位置有关，这与推测解码的核心洞察相矛盾——有些 token 更简单，可以被小模型预测。

**动态树结构：**

不同于 EAGLE-1 或 Medusa 使用固定的树形状，EAGLE-2 会在每一步生成时评估当前 token 的置信度：
- **确定性场景**: 树会变得很深，进行长路径推测
- **不确定场景**: 树会变宽，探索更多可能的分支

**校准性验证：**

EAGLE-2 的关键创新在于发现 **EAGLE 的草稿模型是良好校准的**：草稿模型的置信度分数（概率）是接受率的良好近似。这使得在不运行目标 LLM 的情况下，动态调整草稿树结构成为可能。

**优势分析：**

EAGLE-2 的最大优势在于其**自适应性**。在处理代码或专业文献等确定性强的任务时，其加速比可达 3x 以上；而在闲聊等发散性任务中，它通过变宽的树结构有效维持了接受率，避免了频繁的回退。

### 2.3 EAGLE-3: Scaling up Inference Acceleration via Training-Time Test

**核心贡献：**

EAGLE-3 是该系列的集大成者，核心贡献集中在：

1. **训练时测试**: 模拟多步生成过程进行训练，使草稿模型能充分受益于扩大训练数据。
2. **多层特征融合**: 不再仅依赖顶层特征，而是融合低、中、高层特征，捕获更丰富的语义信息。
3. **直接 Token 预测**: 放弃特征预测约束，直接预测 token，提供完全的灵活性。

**关键洞察：**

论文发现，虽然现代 LLM 通过扩大训练数据来提升模型能力，但 EAGLE 从增加训练数据中获得的改进有限。原因在于：

EAGLE 的损失函数包含两部分：特征预测损失 $l_{fea}$ 和 token 预测损失 $l_{token}$。特征预测约束限制了草稿模型的表达能力，使其难以从增加的数据中受益。

**解决方案：**

EAGLE-3 引入 **Training-Time Test**：
- 移除特征预测损失 $l_{fea}$
- 在训练过程中模拟多步生成
- 使用多层特征融合替代仅依赖顶层特征

**Scaling Law 发现：**

EAGLE-3 首次观察到：**随着草稿模型训练数据量的增加，加速比呈比例增长**。这一 Scaling 行为在原始 EAGLE 架构中从未被观察到。

**词表优化：**

针对 Llama-3 的 128k 词表，EAGLE-3 通过数据驱动方法识别并保留最核心的 32,768 个 token。这不仅减小了模型大小，还将 Softmax 延迟降低了 4 倍。

---

## 三、MEDUSA vs EAGLE 系列对比总结

从架构设计角度分析，MEDUSA 和 EAGLE 系列代表了两种不同的思路：

| 特性 | MEDUSA | EAGLE 系列 |
|------|--------|-----------|
| **预测方式** | 多个独立 MLP 头并行预测 | 特征层自回归预测 |
| **计算开销** | k 个 LM Head 并行计算 | k 次自回归前向传播 |
| **预测依赖** | 各头独立，无依赖 | 后续预测依赖前面的采样结果 |
| **接受率** | 约 60% | 约 80%+ |
| **训练复杂度** | 仅训练头 | 训练轻量 Transformer |

**MEDUSA** 本质上也是一种草稿模型方案，只不过草稿模型是多层 LM Head。每个头独立预测，计算上可以完全并行，但缺乏 token 间的依赖建模。

**EAGLE** 系列实际上修改了模型结构，扩大了输入数据量（引入提前一步的 token 序列），并沿用了自回归机制。从计算角度来看：
- k 次草稿需要 k 个 LM Head 和自回归 head 的计算
- MEDUSA 只需要 k 个并行的 LM Head

EAGLE 系列通过在特征层操作，利用了目标模型的丰富上下文信息，实现了更高的接受率和加速比。

---

## 四、突破瓶颈：云侧扩展的新方向

当 EAGLE-3 达到约 3x 加速后，进一步优化遇到瓶颈。近期的研究从不同角度探索了新的突破方向。

### 4.1 DFlash: Block Diffusion for Flash Speculative Decoding

**核心贡献：**

DFlash 引入了 **扩散模型** 的思想，试图打破传统自回归草稿生成的限制：

1. **并行块生成**: 不同于 EAGLE 的序列生成，DFlash 可以同时生成一整块 token
2. **利用目标模型的隐藏特征**: 将草稿模型作为扩散适配器，高效利用目标模型建模的深度上下文特征

**核心方法：**

DFlash 使用轻量级块扩散模型进行并行草稿：
- 利用目标 LLM 的隐藏特征作为条件上下文
- 在单次前向传播中生成所有 γ 个 token
- 训练简单的 mask token 嵌入来编码 MTP 位置的有意义输入

**自回归 vs 扩散草稿对比：**

| 方式 | 草稿开销 | 特点 |
|------|----------|------|
| 自回归 | $T_{draft} = \gamma \cdot t_{step}$ | 线性增长，受限于浅层架构 |
| 扩散 | $T_{draft} = t_{step}$ | 恒定开销，与 γ 无关 |

**优势分析：**

DFlash 在 Qwen3-8B 上实现了最高 6.1x 加速，比 EAGLE-3 快 2.5x。扩散草稿的关键优势在于：
- 单次前向传播生成多个 token
- 避免了自回归的累积误差
- 更适合高吞吐量场景

### 4.2 P-EAGLE: Parallel-Drafting EAGLE with Scalable Training

**核心贡献：**

P-EAGLE 将 EAGLE 从自回归生成转变为并行多 token 预测：

1. **可学习共享隐藏状态**: 通过 $h_{shared}$ 替代缺失的隐藏向量
2. **可扩展的长上下文训练**: 通过注意力掩码预计算和序列分区技术，支持 20K token 训练

**挑战与解决：**

并行草稿预测面临内存扩展挑战——有效序列长度随并行预测位置数线性增长。P-EAGLE 的解决方案：
- **Amortized Mask Construction**: 预计算注意力掩码
- **Sequence Partitioning**: 将单个序列分段进行梯度累积

**架构创新：**

P-EAGLE 引入可学习参数：
- **$h_{shared}$**: 替代 MTP 位置缺失的隐藏向量
- **Mask Token Embedding**: 替代未知的前置 token

理论分析表明，注意力机制本身就编码了足够的位置信息，无需位置特定的隐藏状态。

### 4.3 Speculative Speculative Decoding (SSD)

**核心贡献：**

这篇论文提出了一个非常有意思的"套娃"思路：**推测中再套一层推测**。

**核心思想：**

在验证进行时，草稿模型预测可能的验证结果，并提前准备相应的推测：
- 如果实际验证结果在预测集合中，可以立即返回推测
- 完全消除草稿开销

**三大挑战：**

1. **预测验证结果**: 需要预测接受的 token 数量和 bonus token
2. **接受率与预测准确性的权衡**: 需要平衡两个目标
3. **失败处理策略**: 批量大小大时失败更频繁

**Saguaro 算法：**

- 使用最可能的草稿 logits 预测 bonus token，准确率达 90%
- 开发采样算法平衡预测准确性和推测质量
- 根据批量大小选择最优回退策略

**结果：** 平均比最强推测解码基线快 30%，比自回归解码快 5x。

---

## 五、云侧优化方向总结

从上述工作可以归纳出云侧推测解码的主要优化方向：

### 5.1 扩大参数规模

P-EAGLE 使用 4 层 Transformer 架构，比单层 EAGLE-3 接受长度高 46%。在云侧场景，内存和算力相对充裕，适当增加草稿模型容量是有效的优化方向。

### 5.2 并行化草稿生成

DFlash 和 P-EAGLE 都致力于消除自回归草稿的串行瓶颈：
- **DFlash**: 使用扩散模型，单次前向生成整块 token
- **P-EAGLE**: 通过共享隐藏状态，实现并行多 token 预测

### 5.3 异步流水线

SSD 展示了将草稿和验证重叠执行的潜力。在分布式部署场景，草稿模型位于独立硬件上，可以与验证并行执行。

### 5.4 适用场景

| 优化方向 | 适用场景 | 加速效果 |
|----------|----------|----------|
| 扩大参数规模 | 云侧、高吞吐量 | 1.1-1.4x |
| 并行草稿生成 | 云侧、大批量 | 1.5-2.5x |
| 异步流水线 | 分布式部署 | 1.3x |

---

## 六、细节优化：词表与动态策略

### 6.1 Speculative Decoding with a Speculative Vocabulary (SpecVocab)

**核心问题：**

现代 LLM 的词表越来越大（如 Qwen3 的 152K），计算输出分布的开销成为草稿阶段的主要瓶颈。在 EAGLE 框架中，大部分草稿时间花在计算目标词表上的输出分布。

**现有方案的问题：**

FR-Spec、EAGLE-3、VocabTrim 都使用固定词表子集来减少投影延迟。但当目标 token 超出子集时，当前和后续草稿 token 都会被拒绝，抵消了加速效果。

**SpecVocab 方法：**

提出**词表推测**作为固定词表的替代方案：

1. **词表排序**: 使用草稿模型最终隐藏状态 $h_t$ 计算词表排名
2. **候选选择**: 选择 top-k 个上下文相关的词表子集
3. **输出计算**: 仅计算候选词表的 logits

$$K_t = \text{top-k}(s_t, k)$$
$$z'_t = U_{K_t} h_t$$

**优势：**

SpecVocab 是上下文感知的，可以使用比固定方法（32K）更小的子集（通常 2048），同时保持更好的覆盖率。实验表明，SpecVocab 比 EAGLE-3 实现了更高的接受长度，吞吐量提升 8.1%。

### 6.2 Draft Model Knows When to Stop: Self-Verification for Long-Form Generation

**核心问题：**

传统推测解码方法使用固定长度策略提出草稿，假设目标模型会顺利接受草稿 token。但实际中：
- Oracle 草稿长度变化很大
- 固定长度策略难以满足这一要求
- 在复杂推理和长文本生成场景，这种差异更加严重

**关键发现：**

通过理论和实证分析，论文发现草稿模型和目标模型之间的差异可以用草稿模型的预测熵来近似：
- **高熵** 表示草稿 token 接受率低
- **低熵** 表示接受率高

**SVIP 方法：**

提出 **Self-Verification Length Policy**，一种无需训练的动态长度策略：

$$\beta \geq 1 - \frac{1}{\sqrt{2}} \sqrt{H_{q,p} - H_q}$$

使用草稿模型熵 $H_q$ 作为代理来近似接受率下界。当熵超过阈值时，草稿模型停止生成并启动验证。

**效果：**

- MT-Bench 8K 上下文：17% 加速
- QwQ 长文本推理：22% 加速
- 与 EAGLE-2 结合：额外 13% 加速

---

## 七、总结与展望

### 7.1 技术演进路线

推测解码技术经历了三个阶段的演进：

1. **基础框架阶段**: Speculative Sampling → MEDUSA
   - 确立了"草稿-验证"范式
   - MEDUSA 展示了无需独立草稿模型的可能性

2. **特征层优化阶段**: EAGLE → EAGLE-2 → EAGLE-3
   - 特征层自回归提供更高接受率
   - 动态树结构适应上下文
   - Training-Time Test 实现 Scaling Law

3. **突破瓶颈阶段**: DFlash、P-EAGLE、SSD
   - 扩散模型打破自回归限制
   - 并行草稿消除串行开销
   - 异步流水线最大化硬件利用

### 7.2 关键洞察

1. **特征层优于词表层**: 特征序列比 token 序列更有规律，预测更准确
2. **上下文自适应**: 接受率高度依赖上下文，静态策略难以最优
3. **并行化是关键**: 草稿生成的串行性是最终瓶颈
4. **细节决定成败**: 词表优化、动态长度策略在极限场景下至关重要

### 7.3 未来方向

1. **端云协同**: 如何在端侧有限资源下实现高效推测解码
2. **长文本优化**: CoT 推理模型的长时间生成需要更激进的优化
3. **多模态扩展**: 推测解码思想是否可以扩展到多模态生成

---

## 参考文献

1. Cai et al., "MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", ICML 2024
2. Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", ICML 2024
3. Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees", ACL 2024
4. Li et al., "EAGLE-3: Scaling up Inference Acceleration via Training-Time Test", 2025
5. Chen et al., "DFlash: Block Diffusion for Flash Speculative Decoding", 2026
6. Hui et al., "P-EAGLE: Parallel-Drafting EAGLE with Scalable Training", 2026
7. Kumar et al., "Speculative Speculative Decoding", 2026
8. Williams et al., "Speculative Decoding with a Speculative Vocabulary", 2026
9. Zhang et al., "Draft Model Knows When to Stop: Self-Verification Speculative Decoding for Long-Form Generation", EMNLP 2024