#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            text = f.read()
        
        self.chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.char2idx[ch] for ch in text]
        self.seq_length = 30
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(input_seq), torch.tensor(target_seq)


