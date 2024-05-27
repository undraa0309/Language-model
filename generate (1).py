#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
from model import CharRNN, CharLSTM
from dataset import char_tensor, all_characters
import argparse

def generate(model, seed_characters, temperature=1.0, predict_len=100):
    model.eval()
    hidden = model.init_hidden(1).to(next(model.parameters()).device)
    seed_input = char_tensor(seed_characters).unsqueeze(0).to(next(model.parameters()).device)
    predicted = seed_characters

    with torch.no_grad():
        for c in seed_input.squeeze(0):
            _, hidden = model(c.unsqueeze(0).unsqueeze(0), hidden)

        inp = seed_input[:, -1]

        for _ in range(predict_len):
            output, hidden = model(inp.unsqueeze(0), hidden)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_i]
            predicted += predicted_char
            inp = char_tensor(predicted_char).unsqueeze(0).to(next(model.parameters()).device)

    return predicted

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using a trained RNN/LSTM model.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--seed', type=str, required=True, help='Seed characters to start the generation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for softmax')
    parser.add_argument('--length', type=int, default=100, help='Length of text to generate')
    parser.add_argument('--model_type', type=str, choices=['RNN', 'LSTM'], default='LSTM', help='Type of model to use')
    args = parser.parse_args()

    checkpoint = torch.load(args.model)
    model_type = checkpoint['model_type']
    hidden_size = checkpoint['n_hidden']
    n_layers = checkpoint['n_layers']
    chars = checkpoint['tokens']

    if model_type == 'LSTM':
        model = CharLSTM(len(chars), hidden_size, n_layers)
    else:
        model = CharRNN(len(chars), hidden_size, n_layers)

    model.load_state_dict(checkpoint['state_dict'])
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    samples = generate(model, args.seed, args.temperature, args.length)
    print(samples)




