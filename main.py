import json
import torch
import random
from torch import nn, optim
from model import Transformer
from dataset import get_dataset, cmn_words, eng_words, seq_len

torch.manual_seed(9527), random.seed(7527)

if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")


def train(model, dataloader, backward, epochs=10):
    losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for source, target in dataloader:
            source, target = source.to(device), target.to(device)
            logits = model(source, target[:, :-1])
            loss = backward(logits.view(-1, logits.size(-1)), target[:, 1:].reshape(-1))
            loss = loss.cpu().item()
            losses.append(loss)
            total_loss += loss
        print(f"Epoch {epoch+1:>2}/{epochs}, Loss: {total_loss / len(dataloader)}")
    return losses


@torch.no_grad()
def eval(model, dataloader, print_result):
    model.eval()
    for x, Y in dataloader:
        x, y = x.to(device), torch.ones(1, 1, dtype=int, device=device)
        for _ in range(seq_len - 1):
            logits = model(x, y).argmax(dim=-1)
            y = torch.concat([y, logits[:, -1:]], dim=-1)
            if y[0, -1].item() == 2:
                break
        print_result(
            [index for index in x.view(-1).tolist() if index > 2],
            [index for index in Y.view(-1).tolist() if index > 2],
            [index for index in y.view(-1).tolist() if index > 2],
        )


train_loader, test_loader = get_dataset()
input_size, output_size = len(cmn_words) + 3, len(eng_words) + 3

model = Transformer(input_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)


def backward(logits, target):
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


losses = train(model, train_loader, backward)
torch.save(model.state_dict(), "./data/model.pth")

with open("./data/losses.json", "w", encoding="utf-8") as f:
    json.dump(losses, f, indent=4)


def print_result(source, target, output):
    print("source:", " ".join(cmn_words[i - 3] for i in source))
    print("target:", " ".join(eng_words[i - 3] for i in target))
    print("output:", " ".join(eng_words[i - 3] for i in output))
    print()


eval(model, test_loader, print_result)
