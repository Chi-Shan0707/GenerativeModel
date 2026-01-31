# Image GPT

[link](https://openai.com/index/image-gpt/)

# Notes

这是一份为您整合了之前所有问答的完整笔记，涵盖了论文 **Generative Pretraining from Pixels** 的 **Introduction** 与 **Approach** 部分。

特别针对您关注的评估手段（Fine-tuning & Linear Probe）进行了精炼的补充说明。

---

# Note: Generative Pretraining from Pixels

## 1. Introduction (Overview)

**核心思想**：借鉴自然语言处理（NLP）中无监督预训练（Unsupervised Pre-training）的成功经验（如 GPT, BERT），探究是否可以使用同样的 **Transformer** 架构，在不利用任何二维图像先验知识（如 CNN 的卷积操作）的情况下，直接通过**像素级预测**来学习高质量的图像表征。

* **背景**：NLP 领域通过在大规模文本上进行自监督学习（Self-supervised learning）取得了巨大成功，而计算机视觉（CV）领域近期主要依赖有监督学习或利用图像特定结构的自监督方法。
* **方法简述**：将图像视为一维像素序列（1D Sequence of pixels），训练一个序列 Transformer 模型进行预测。
* **结论**：即使没有 2D 空间结构的归纳偏置（Inductive Bias），该模型通过 **Linear Probe** 和 **Fine-tuning** 评估，在 CIFAR-10 和 ImageNet 上均展示了强大的特征提取能力，甚至能匹敌有监督模型。

---

## 2. Approach (Methodology Details)

本节详细阐述模型是如何进行预训练（Pre-training）以及如何评估其表征质量的。

### 2.1 Pre-training Objectives (预训练目标)

论文采用了两种不同的预训练模式（二选一），将图像  视为像素序列 。

#### **A. Auto-Regressive (AR) Objective**

* **定义**：类似于 GPT，基于序列中之前的像素预测下一个像素。
* **数学表达**：最大化似然函数 。
* **Raster Order (栅格顺序)**：
* **概念**：为了将二维图像转化为一维序列以进行 AR 预测，模型采用了标准的栅格顺序（即从左到右，从上到下扫描）。
* **意义**：这人为地定义了像素的排列先后，确定了条件概率的计算路径。



#### **B. BERT Objective**

* **定义**：类似于 BERT 的掩码语言模型（Masked Language Modeling）。随机采样图像中的一部分位置索引 （Mask），掩盖其像素值，让模型根据上下文重建这些被掩盖的像素。
* **数学表达**：。
* **Zero out (置零)**：在输入处理阶段，对于属于  集合的位置，其对应的输入像素值被直接设置为 0，以此作为“完形填空”的题目。

> **💡 核心概念澄清：Permutation Invariance (排列不变性)**
> * **您的疑问**：既然按栅格输入，为何强调“打乱顺序”的特性？
> * **解释**：这是为了强调 Transformer 与 CNN 的**本质区别**。
> * **CNN**：具有强烈的空间归纳偏置，假设相邻像素必相关。如果打乱像素，CNN 就失效了。
> * **Transformer (BERT模式)**：如果不考虑人为强加的 Raster Order（即在 BERT 模式下），模型是 **Permutation Invariant** 的。这意味着，即使输入序列被随机打乱（只要位置编码 Position Embeddings 随之移动），模型对像素间关系的理解不受影响。
> * **结论**：这证明了模型不是靠“死记硬背”固定的空间邻域，而是在训练中**自己学会了**（Learned at train time）图像的空间结构关系。
> 
> 
> 
> 

---

### 2.2 Architecture (模型架构)

模型采用了 **Transformer Decoder** 架构（基于 GPT-2 formulation）。

#### **核心组件与操作**

1. **Input Processing**：
* 输入序列被转化为  维的 Embeddings（包含像素值 Embedding 和位置 Embedding）。


2. **Transformer Block**：
* **Pre-activation Layer Norms**：**Layer Norm** 被放置在 Attention 和 MLP 操作**之前**。
* **公式**：




* **Logits**：模型最后一层输出的原始数值，用于计算分类的概率分布。在 BERT 训练中，计算 Loss 时会忽略非掩码位置的 Logits。



#### **AR 与 BERT 的架构统一性**

同一个网络架构通过改变**注意力掩码（Attention Mask）**来适应不同任务：

* **AR 模式**：应用标准的上三角掩码（Upper Triangular Mask），确保位置  只能看到  的信息。
* **BERT 模式**：不使用注意力掩码（No attention logit masking），允许看到全局上下文，但输入端部分像素被 **Zero out**。

---

### 2.3 & 2.4 Evaluation Strategy (评估策略)

为了验证无监督预训练学到的特征是否有效，作者使用了两种评估手段。**这两种手段分别从“现状”和“潜力”两个维度验证了模型的能力。**

#### **手段 1: Linear Probing (线性探测) — 测“现状” (Static Feature Quality)**

* **操作**：**冻结**预训练模型参数，仅训练最后一层线性分类器。
* **核心逻辑**：如果特征够好，简单的线性层也能分得准。
* **评估意义**：
* 评估预训练特征的**线性可分性 (Linear Separability)**。
* 排除模型架构微调带来的干扰，纯粹衡量特征提取器（Feature Extractor）目前的质量。



#### **手段 2: Fine-Tuning (微调) — 测“潜力” (Dynamic Transferability)**

* **操作**：**解冻**全模型参数，在下游任务上进行有监督训练。
* **核心逻辑**：如果底子够好，学新东西就应该又快又好。即只用在微调时加一个小的正则化项，即可在小数据集上训练出超高准确率
* **评估意义**：
* **Favorable Initialization (有利初始化)**：评估预训练参数是否提供了一个离最优解更近的起点。
* **Regularizer (正则化作用)**：结合 **Early Stopping**，评估预训练是否能作为约束，防止模型在小数据集上过拟合。


* **特征提取 (Pooling)**：（对序列维度取平均，得到整图特征向量）。
* **Joint Objective (联合目标)**：微调时同时优化分类误差和生成误差（）效果最佳。

---
这是一份为您定制的关于 **Methodology** 和 **Training Details** 的补充笔记，格式与之前保持一致。

---

## 3. Methodology & Training Details

本节重点阐述了为了将 Transformer 应用于高维图像数据所采取的工程化手段（Context Reduction），以及确保大规模模型能够稳定收敛的训练策略。

### 3.1 Context Reduction: The 9-bit Color Palette (上下文缩减：9位调色板)

由于 Transformer 的注意力机制（Self-Attention）对序列长度呈二次方复杂度（），直接处理标准 RGB 像素序列（长度极长）在计算上是不可行的 。作者采用了一种基于聚类的降维策略。

* **The Palette Strategy (调色板策略)**：
* **操作 (Operation)**：使用 **k-means** 聚类算法，将所有  像素值聚类为  个簇（Clusters） 。


* **9-bit 的含义**：由于 ，这 512 种颜色可以由一个 9 位的整数（ID）唯一表示 。


* **结果 (Result)**：输入序列长度缩短了 **3倍**。原本需要 3 个数值 (R, G, B) 表示一个像素，现在只需要 1 个 Token ID 。




* **Invariance Properties (不变性特征)** ：


* **Keeps Spatial Invariance (保持空间不变性)**：颜色映射规则在整张图片上是统一的，不依赖于像素的坐标位置。
* **Breaks Color Channel Permutation Invariance (破坏颜色通道排列不变性)**：
* 原始 RGB 通道是独立的（我们可以单独交换 R 和 B 通道）。
* 聚类后，RGB 三个数值被“融合”成了一个单一的 Cluster ID。一旦变成 ID，就无法再拆分或独立操作 R/G/B 分量，因此破坏了通道间的对称性。





### 3.2 Training Dynamics (训练参数与策略)

为了训练这种非传统的图像生成模型，作者对标准的训练参数进行了针对性的调整。

#### **A. Optimizers (优化器)**

1. **Adam (for Pre-training & Fine-tuning)**
* **用途**：用于模型的主干预训练（Pre-training）和全模型微调（Fine-tuning） 。


* **关键调整 ()**：
* 标准 Adam 的  默认为 0.999。
* 作者发现默认值会导致训练过程中出现不可恢复的 Loss Spikes (损失激增)，因此将其调低至 **0.95** 以增加稳定性 。




* **Batch Size**：超大模型 (iGPT-XL) 为 **64**，其他模型为 **128** 。




2. **SGD (for Linear Probing on ImageNet)**
* **用途**：专用于在 ImageNet 上进行 **Linear Probe**（线性探测）评估 。


* **配置**：使用带有 **Momentum=0.9** 的 SGD，并配合较高的学习率（如 30, 10, 3） 。


* **原因**：为了遵循文献中的标准评估设置（Follow recent literature），以便与其他自监督方法进行公平对比 。





#### **B. Schedulers & Regularization (调度与正则化)**

1. **Cosine Learning Rate Schedule (余弦学习率调度)**
* **机制**：学习率首先进行 1 个 epoch 的 **Warm-up**（热身），然后按照余弦曲线衰减至 0 。


* **应用场景**：Pre-training 阶段和 Linear Probing 阶段 。


* **目的**：在训练初期提供较大的探索空间，后期通过平滑衰减实现精细收敛。


2. **Early Stopping (早停法)**
* **机制**：一旦验证集准确率（Validation Accuracy）达到峰值，立即停止训练 。


* **应用场景**：**Fine-tuning (微调)** 阶段 。


* **正则化作用**：由于下游任务（如 CIFAR-10）数据量较小，全模型微调极易过拟合。Early Stopping 在此处充当了强力的 **Regularizer**，防止模型死记硬背训练数据 。


* 
*注意*：在微调阶段不使用 Cosine Schedule，而是直接依赖 Early Stopping 。