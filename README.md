# Attention Is All You Need

Transformer 模型架构作为大语言模型（LLM）的基石，起源于 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 这篇论文。随着 LLM 的大热，吸引了很多研究者复现这篇论文。我找了一些来学习，但那些代码实现要么太过老旧跑不起来，要么太过复杂看不明白。于是我参考 [PyTorch 官方实现](https://github.com/pytorch/pytorch/blob/0d9c95cd7ee299e2e8c09df26d395be8775b506b/torch/nn/modules/transformer.py#L57)等资料，搞出这个**忠于原文、聚焦核心**的实现，舍弃一切预处理、工程优化等无关内容，力求简单明了，以便学习理解。

我重写 PyTorch 官方代码，压缩 10 倍，以 **124 行代码**涵盖模型的全部细节，并且代码结构尽力贴近原文描述。原来很多细节在原始论文中没有提及，在代码实现中才能看到。我用中译英验证了其有效性和正确性。由于中英两种语言差异较大，中译英是较为困难的任务，我在 [Seq2Seq(RNN + Attention)](https://github.com/ngolin/Seq2seq/) 上难以达到的效果，在 Transformer 上却轻松达到，体现了 Transformer 的强大。

## 一、快速开始

**模型定义：**[model.py](./model.py)，需配合原文阅读代码无额外注释，查看模型定义：

> ```bash
> $ python model.py
> ```

**模型验证：**[main.ipynb](./main.ipynb)，中译英验证模型实现是否正确，查看数据概要：

> ```bash
> $ python -m dataset
> ```

## 二、重要解读

### 1. 注意力机制

注意力（Attention）是 Transformer 架构的核心机制，细节较多，但本质上还是全连接（FNN）的变体。我们将看到，强如 Transformer, 细如 Attention, 也没有脱离 FNN 的底色。实际上 FNN 在模型设计中随处可见，以 FNN 为基础，RNN 加入循环逻辑，LSTM 通过门控机制解决长程依赖问题，GRU 进一步简化计算，但其计算单元还是 FNN；CNN 引入卷积计算，但仍然是层级结构；ResNet 增加跳跃连接，解决深层网络退化问题，但还是 FNN 的信息传递模式。注意力机制本质上也是加权求和和非线性变换，我们将从三个层面帮助理解：

1. 如何来理解注意力机制的本质？
2. 为什么要对权重缩放并归一化？
3. 注意力权重掩码有什么作用呢？

万变不离其宗，无论 Transformer/Attention 设计得多么精炒，我们依然可以看到那个简单而朴素的 FNN 影子。这是深度学习的魅力所在，更是一个强大的启示。当我们遇到新的问题或面临全新领域时，或许也要回到 FNN 的基础上进行求解和创新，通过拆解其层级结构，调整其连接方式，使 FNN 更有效地表征和适配数据；通过权重矩阵的多信号感知和激活函数的非线性变换，使 FNN 更高效地聚合和传递信息，从而赋予新的机制，解决新的问题。

#### 1.1 如何来理解注意力机制的本质？

注意力的核心计算逻辑同样是 $\mathbf{Y} = \mathbf{W} \cdot \mathbf{X}$, 只不过细节上有些调整，也换了不同的名称 $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$。如果 $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ 来自同一个 Sequence 的线性变换，则称为**自注意力**；如果 $\mathbf{Q}$ 来自 $\mathbf{T} = [\mathbf{T}_1, \mathbf{T}_2, \dots, \mathbf{T}_T]$, 而 $\mathbf{K}$, $\mathbf{V}$ 来自不同的 $\mathbf{S} = [\mathbf{S}_1, \mathbf{S}_2, \dots, \mathbf{S}_S]$, 则称为**交叉注意力**；在自回归生成模型中，模型逐个生成 Token, 因此在训练时当前 Token 不能之后的 Tokens 加权求和，称为**因果注意力**。

|                                                                                                                                        **自注意力**                                                                                                                                         |                                                                                                                                                                                **因果自注意力**                                                                                                                                                                                 |                                                                                                                                                                      **交叉注意力**                                                                                                                                                                       |
| :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                             $\mathbf{Q} = \text{Linear}(\mathbf{S})$<br/>$\mathbf{K} = \text{Linear}(\mathbf{S})$<br/>$\mathbf{V} = \text{Linear}(\mathbf{S})$                                                                              |                                                                                                                       $\mathbf{Q} = \text{Linear}(\mathbf{T})$<br/>$\mathbf{K} = \text{Linear}(\mathbf{T})$<br/>$\mathbf{V} = \text{Linear}(\mathbf{T})$                                                                                                                        |                                                                                               $\mathbf{Q} = \text{Linear}(\mathbf{T} ^ \prime)$<br/>$\mathbf{K} = \text{Linear}(\mathbf{S} ^ \prime)$<br/>$\mathbf{V} = \text{Linear}(\mathbf{S} ^ \prime)$                                                                                               |
| $\mathbf{S} = \text{SelfAttention}(\mathbf{S}) = \mathbf{W} ^ \prime \cdot \mathbf{V}$<br/>$\mathbf{W} ^ \prime = \text{Softmax}(\frac{\mathbf{W} = \mathbf{Q} \cdot \mathbf{K} ^ T}{\sqrt{d = 512}})$<br/>$W _ {ij} = \mathbf{Q} _ i \cdot \mathbf{K} _ j, 1 \leqslant (i, j) \leqslant S$ | $\mathbf{T} = \text{CasualSelfAttention}(\mathbf{T}) = \mathbf{W} ^ \prime \cdot \mathbf{V}$<br/>$\mathbf{W} ^ \prime = \text{Softmax}(\frac{\mathbf{W} = \mathbf{Q} \cdot \mathbf{K} ^ T}{\sqrt{d = 512}} + \mathbf{M})$ <br/> $W _ {ij} = \mathbf{Q} _ i \cdot \mathbf{K} _ j, 1 \leqslant (i, j) \leqslant T$ <br/> $M _ {ij} = 0 \text{ if } i < j \text{ else } {-\infty}$ | $\mathbf{T} ^ \prime = \text{CrossAttention}(\mathbf{T} ^ \prime, \mathbf{S} ^ \prime) = \mathbf{W} ^ \prime \cdot \mathbf{V}$<br/>$\mathbf{W} ^ \prime = \text{Softmax}(\frac{\mathbf{W} = \mathbf{Q} \cdot \mathbf{K} ^ T}{\sqrt{d = 512}})$<br/>$W _ {ij} = \mathbf{Q} _ i \cdot \mathbf{K} _ j, 1 \leqslant i \leqslant T, 1 \leqslant j \leqslant S$ |
|                                                                                                                      $\mathbf{S} ^ \prime = \text{Linear}(\mathbf{S})$                                                                                                                      |                                                                                                                                                                $\mathbf{T} ^ \prime = \text{Linear}(\mathbf{T})$                                                                                                                                                                |                                                                                                                                            $\mathbf{T} ^ {\prime \prime} = \text{Linear}(\mathbf{T} ^ \prime)$                                                                                                                                            |

在 Transformer 架构中，编码器有一种自注意力，解码器有一种因果自注意力和交叉注意力。每种注意力都有 6 层 8 头，共 48 个注意力。所谓的 6 层，其实就是重复 6 次；正因为不止 1 头，才叫**多头注意力**。所谓的 8 头，就是把词嵌入的 512 维分为 8 组每组 64 维分别进行注意力加权求和，可以理解为 8 头注意力支持对一个 Token 多达 8 个不同的语义分别应用注意力机制。

```mermaid
---
config:
  sankey:
    width: 1000
---
sankey

S,S1,512
S,S2,512
S,S3,512

S1,H1,64
S2,H1,64
S3,H1,64

S1,H2,64
S2,H2,64
S3,H2,64

S1,H3,64
S2,H3,64
S3,H3,64

S1,H4,64
S2,H4,64
S3,H4,64

S1,H5,64
S2,H5,64
S3,H5,64

S1,H6,64
S2,H6,64
S3,H6,64

S1,H7,64
S2,H7,64
S3,H7,64

S1,H8,64
S2,H8,64
S3,H8,64

H1,Attention(H1),192
H2,Attention(H2),192
H3,Attention(H3),192
H4,Attention(H4),192
H5,Attention(H5),192
H6,Attention(H6),192
H7,Attention(H7),192
H8,Attention(H8),192

Attention(H1),S1',64
Attention(H2),S1',64
Attention(H3),S1',64
Attention(H4),S1',64
Attention(H5),S1',64
Attention(H6),S1',64
Attention(H7),S1',64
Attention(H8),S1',64

Attention(H1),S2',64
Attention(H2),S2',64
Attention(H3),S2',64
Attention(H4),S2',64
Attention(H5),S2',64
Attention(H6),S2',64
Attention(H7),S2',64
Attention(H8),S2',64

Attention(H1),S3',64
Attention(H2),S3',64
Attention(H3),S3',64
Attention(H4),S3',64
Attention(H5),S3',64
Attention(H6),S3',64
Attention(H7),S3',64
Attention(H8),S3',64

S1',S',512
S2',S',512
S3',S',512
```

所谓的加权求和，就是一个 Sequence 转换成另一个 Sequence, 新 Sequence 的每个 Token 为旧 Sequence 所有 Tokens 加权求和所得，也就是每个 Token 都融合了所有 Tokens 的信息。由于每个 Token 都要依赖上下文才能明确自身语义，加权求和正是出于这个考虑，每个 Token 都可以融合其他所有 Token 的信息来明确自身的语义，不同大小的权重分配代表不同程度的依赖。

```mermaid
block
columns 36
  S1 space:4
  S2 space:4
  S3 space:4
  S4 space:4
  T1 space:4
  T2 space:4
  T3 space:4
  T4

  space:144

  S1' space:4
  S2' space:4
  S3' space:4
  S4' space:4
  T1' space:4
  T2' space:4
  T3' space:4
  T4'

  space:288

  space:10
  T1" space:4
  T2" space:4
  T3" space:4
  T4"



  S1 --> S1'
  S2 --> S1'
  S3 --> S1'
  S4 --> S1'

  S1 --> S2'
  S2 --> S2'
  S3 --> S2'
  S4 --> S2'

  S1 --> S3'
  S2 --> S3'
  S3 --> S3'
  S4 --> S3'

  S1 --"SelfAttention"--> S4'
  S2 --> S4'
  S3 --> S4'
  S4 --> S4'

  T1 --> T1'

  T1 --> T2'
  T2 --> T2'

  T1 --> T3'
  T2 --> T3'
  T3 --> T3'

  T1 --"CasualSelfAttention"--> T4'
  T2 --> T4'
  T3 --> T4'
  T4 --> T4'

  S1' --> T1"
  S2' --> T1"
  S3' --> T1"
  S4' --> T1"
  T1' --> T1"

  S1' --> T2"
  S2' --> T2"
  S3' --> T2"
  S4' --> T2"
  T2' --> T2"

  S1' --> T3"
  S2' --> T3"
  S3' --> T3"
  S4' --> T3"
  T3' --> T3"

  S1' --> T4"
  S2' --> T4"
  S3' --"CrossAttention"--> T4"
  S4' --> T4"
  T4' --> T4"
```

#### 1.2 为什么要对权重缩放并归一化？

#### 1.3 注意力权重掩码有什么作用呢？

### 2. LayerNorm 正则化

### 3. 词嵌入与位置编码

## 三、代码实现

## 四、参考资料

- 论文原文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 论文解读：[Transformer 论文逐段精读](https://www.youtube.com/watch?v=nzqlFIcCSWQ)
- 官方实现：[pytorch/torch/nn/modules/transformer.py](https://github.com/pytorch/pytorch/blob/0d9c95cd7ee299e2e8c09df26d395be8775b506b/torch/nn/modules/transformer.py#L57)
- 官方示例：[examples/language_translation/src/model.py](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/language_translation/src/model.py#L28)
- nanoGPT：[nanoGPT](https://github.com/karpathy/nanoGPT/blob/93a43d9a5c22450bbf06e78da2cb6eeef084b717/model.py#L52-L76)
