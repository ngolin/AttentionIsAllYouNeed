import torch
from . import cmn_eng, cmn_words, eng_words, get_dataset

torch.manual_seed(9527)
print(len(cmn_eng), len(cmn_words) + 3, len(eng_words) + 3)
for train, test in zip(*get_dataset()):
    print(*map(lambda x: x.shape, train))
    print(*test, sep="\n")
    break
