#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
from model import VanillaRNN, LSTM
from dataset import TextDataset

def generate_text(model, start_char, char_to_idx, idx_to_char, hidden, length, temperature):
    model.eval()
    input_char = torch.tensor([char_to_idx[start_char]]).unsqueeze(0).to(next(model.parameters()).device)
    generated_text = start_char

    for _ in range(length):
        if isinstance(model, VanillaRNN):
            output, hidden = model(input_char, hidden)
        else:
            hidden, cell = hidden
            output, hidden, cell = model(input_char, hidden, cell)
            hidden = (hidden, cell)

        output = output.squeeze().div(temperature).exp()
        probabilities = F.softmax(output, dim=0)
        top_i = torch.multinomial(probabilities, 1)[0]
        char = idx_to_char[top_i.item()]

        generated_text += char
        input_char = torch.tensor([top_i.item()]).unsqueeze(0).to(next(model.parameters()).device)

    return generated_text

def main():
    # Load model
    model_path = 'best_model.pth'
    model_type = 'LSTM'  # or 'VanillaRNN'
    hidden_size = 128
    num_layers = 2

    with open('shakespeare.txt', 'r') as f:
        text = f.read()

    dataset = TextDataset(text, seq_length=100)
    vocab_size = dataset.vocab_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'LSTM':
        model = LSTM(vocab_size, hidden_size, num_layers).to(device)
        hidden = (torch.zeros(num_layers, 1, hidden_size).to(device),
                  torch.zeros(num_layers, 1, hidden_size).to(device))
    else:
        model = VanillaRNN(vocab_size, hidden_size, num_layers).to(device)
        hidden = torch.zeros(num_layers, 1, hidden_size).to(device)

    model.load_state_dict(torch.load(model_path))

    # Generate text
    seed_chars = ['H', 'T', 'W', 'A', 'S']
    for seed_char in seed_chars:
        for temp in [0.5, 1.0, 1.5]:
            generated_text = generate_text(model, seed_char, dataset.char_to_idx, dataset.idx_to_char, hidden, length=100, temperature=temp)
            print(f'Seed: {seed_char}, Temp: {temp}\n{generated_text}\n')

if __name__ == '__main__':
    main()


# In[ ]:




