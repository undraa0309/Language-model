#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.seq_length = seq_length
        self.vocab_size = len(self.chars)
        self.data = self.preprocess()

    def preprocess(self):
        return [self.char_to_idx[ch] for ch in self.text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_length]
        target = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(seq), torch.tensor(target)

def create_dataloader(text, seq_length, batch_size):
    dataset = TextDataset(text, seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

