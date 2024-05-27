#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
from model import CharRNN, CharLSTM
from dataset import char_tensor, all_characters
import argparse

def evaluate(model, prime_str='A', predict_len=100, temperature=0.8):
    model.eval()
    hidden = model.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

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




