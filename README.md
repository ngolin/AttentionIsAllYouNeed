# Attention Is All You Need

基于 PyTorch 实现，遵循 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 论文的神经网络机器翻译 (NMT) 模型。本项目旨在提供一个结构清晰、易于理解和修改的 Transformer 架构实现，用于中文到英文的翻译任务。

- 模型定义：[model.py](./model.py)
- 模型训练：[main.ipynb](./main.ipynb)

## 模型架构

```python
Transformer(
  (encoder): Encoder(
    (0-5): 6 x EncoderBlock(
      (attn): AddAndNorm(
        (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (out): Dropout(p=0.1, inplace=False)
        (sub): MultiHeadAttention(
          (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
        )
      )
      (feed): AddAndNorm(
        (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (out): Dropout(p=0.1, inplace=False)
        (sub): FeedForward(
          (0): Linear(in_features=512, out_features=2048, bias=True)
          (1): ReLU()
          (2): Linear(in_features=2048, out_features=512, bias=True)
        )
      )
    )
  )
  (decoder): Decoder(
    (0-5): 6 x DecoderBlock(
      (att1): AddAndNorm(
        (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (out): Dropout(p=0.1, inplace=False)
        (sub): MultiHeadAttention(
          (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
        )
      )
      (att2): AddAndNorm(
        (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (out): Dropout(p=0.1, inplace=False)
        (sub): MultiHeadAttention(
          (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
        )
      )
      (feed): AddAndNorm(
        (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (out): Dropout(p=0.1, inplace=False)
        (sub): FeedForward(
          (0): Linear(in_features=512, out_features=2048, bias=True)
          (1): ReLU()
          (2): Linear(in_features=2048, out_features=512, bias=True)
        )
      )
    )
  )
  (emi): Embedding(2387, 512, padding_idx=0)
  (emo): Embedding(1938, 512, padding_idx=0)
  (lin): Linear(in_features=512, out_features=1938, bias=True)
  (pos): PositionalEncoding()
  (out): Dropout(p=0.1, inplace=False)
)
```

## 参考资料

- 论文原文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 论文解读：[Transformer 论文逐段精读](https://www.youtube.com/watch?v=nzqlFIcCSWQ)
- 官方实现：[pytorch/torch/nn/modules/transformer.py](https://github.com/pytorch/pytorch/blob/0d9c95cd7ee299e2e8c09df26d395be8775b506b/torch/nn/modules/transformer.py#L57)
- 官方示例：[examples/language_translation/src/model.py](https://github.com/pytorch/examples/blob/acc295dc7b90714f1bf47f06004fc19a7fe235c4/language_translation/src/model.py#L28)
- nanoGPT：[nanoGPT](https://github.com/karpathy/nanoGPT/blob/93a43d9a5c22450bbf06e78da2cb6eeef084b717/model.py#L52-L76)
