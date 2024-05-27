#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, n_hidden=256, n_layers=2, drop_prob=0.5):
        super(CharRNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, n_hidden)
        self.rnn = nn.RNN(n_hidden, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        r_output, hidden = self.rnn(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
        return hidden

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, n_hidden=256, n_layers=2, drop_prob=0.5):
        super(CharLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, n_hidden)
        self.lstm = nn.LSTM(n_hidden, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden

