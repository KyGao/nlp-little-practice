# %%
# code changed from Tae Hwan Jung @graykode

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch, target_batch = [], []

    # predict last char
    for seq in seq_data:
        inputs = [word_dict[n] for n in seq[:-1]]
        targets = word_dict[seq[-1]]
        input_batch.append(np.eye(n_class)[inputs])
        target_batch.append(targets)

    return input_batch, target_batch

class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        inputs = X.transpose(0, 1) # X: [n_step, batch_size, n_class]
