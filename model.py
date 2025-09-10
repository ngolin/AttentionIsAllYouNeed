import math
import torch
import torch.nn as nn
from functools import reduce

model_size, hidden_size, n_layer, n_head, dropout = 512, 2048, 6, 8, 0.1


class Transformer(nn.Module):
    def __init__(self, input_size, output_size, *, max_len=1024, padding_idx=0):
        super().__init__()
        self.encoder, self.decoder = Encoder(), Decoder()

        self.emi = nn.Embedding(input_size, model_size, padding_idx)
        self.emo = nn.Embedding(output_size, model_size, padding_idx)
        self.lin = nn.Linear(model_size, output_size)
        self.pos = PositionalEncoding(max_len)
        self.out = nn.Dropout(dropout)
        self.pad = padding_idx

    def forward(self, inputs, outputs):
        x, mx = self.out(self.emi(inputs) + self.pos(inputs)), inputs == self.pad
        y, my = self.out(self.emo(outputs) + self.pos(outputs)), outputs == self.pad
        return self.decoder(self.encoder(x, mx), mx, y, my)


class Encoder(nn.ModuleList):
    def __init__(self):
        super().__init__(EncoderBlock() for _ in range(n_layer))

    def forward(self, x, mx):
        return reduce(lambda x, f: f(x, mx), self, x)


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = AddAndNorm(MultiHeadAttention())
        self.feed = AddAndNorm(FeedForward())

    def forward(self, x, mx):
        m = mx.unsqueeze(-2) | mx.unsqueeze(-1)
        x = self.attn(x, x, x, m=m)
        x = self.feed(x)
        return x


class Decoder(nn.ModuleList):
    def __init__(self):
        super().__init__(DecoderBlock() for _ in range(n_layer))

    def forward(self, x, mx, y, my):
        return reduce(lambda y, f: f(x, mx, y, my), self, y)


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.att1 = AddAndNorm(MultiHeadAttention(casual=True))
        self.att2 = AddAndNorm(MultiHeadAttention())
        self.feed = AddAndNorm(FeedForward())

    def forward(self, x, mx, y, my):
        m1 = my.unsqueeze(-2) | my.unsqueeze(-1)
        m2 = mx.unsqueeze(-2) | my.unsqueeze(-1)
        y = self.att1(y, y, y, m=m1)
        y = self.att2(y, x, x, m=m2)
        y = self.feed(y)
        return y


class AddAndNorm(nn.Module):
    def __init__(self, sub_layer):
        super().__init__()
        self.norm = nn.LayerNorm(model_size)
        self.out = nn.Dropout(dropout)
        self.sub = sub_layer

    def forward(self, x, *args, **kargs):
        y = self.out(self.sub(x, *args, **kargs))
        return self.norm(x + y)


class MultiHeadAttention(nn.ModuleList):
    def __init__(self, casual=False):
        super().__init__(nn.Linear(model_size, model_size) for _ in range(3))
        self.lin, self.casual = nn.Linear(model_size, model_size), casual

    def forward(self, *qkv, m):
        q, k, v = (
            x.view(*x.shape[:-1], n_head, -1).transpose(-3, -2)
            for x in (f(x) for f, x in zip(self, qkv))
        )
        m = m | (~m).triu(1) if self.casual else m
        w = q @ k.mT / math.sqrt(q.size(-1))
        w = w.masked_fill(m.unsqueeze(-3), float("-inf"))
        w = w.softmax(dim=-1).nan_to_num(0)
        v = (w @ v).transpose(-3, -2)
        v = v.reshape(*v.shape[:-2], -1)
        return self.lin(v)


class FeedForward(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(model_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, model_size),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(-1)
        emb = torch.arange(0, model_size, 2, dtype=torch.float)
        div = (-math.log(10000.0) / model_size * emb).exp()
        pe = torch.zeros(max_len, model_size)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[: x.size(-1)]


if __name__ == "__main__":
    from dataset import cmn_words, eng_words, seq_len, pad

    batch_size, input_size, output_size = 32, len(cmn_words) + 2, len(eng_words) + 2
    model = Transformer(input_size, output_size, max_len=seq_len, padding_idx=pad)

    inputs = torch.randint(input_size, (batch_size, seq_len))
    outputs = torch.randint(output_size, (batch_size, seq_len))
    assert model(inputs, outputs).argmax(dim=-1).shape == outputs.shape
    print(model)
