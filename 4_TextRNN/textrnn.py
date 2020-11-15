# %%
# code changed from Tae Hwan Jung @graykode

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        words = sen.split() # space tokenizer
        inputs = [word_dict[n] for n in words[:-1]] # create (1~n-1) as input
        targets = word_dict[words[-1]] # create (n) as target, we usually call this 'casual language model'

        # inputs and targets are int list / scalar to represent one word
        input_batch.append(np.eye(n_class)[inputs]) # word dict to one-hot vec
        target_batch.append(targets)

    return input_batch, target_batch

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):
        # print("X", X)

        X = X.transpose(0, 1) # X: [n_step, batch_size(=3), n_class(=7)]
        outputs, hidden = self.rnn(X, hidden)
        # outputs: [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden: [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b # model: [batch_size, n_class]
        return model

if __name__ == '__main__':
    n_step = 2 # num of cells (=num of steps)
    n_hidden = 5 # num of hidden units in one cell

    sentences = ['i like dog', 'i love coffee', 'i hate milk']

    word_list = ' '.join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)    # 7
    batch_size = len(sentences) # 3

    model = TextRNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()

        # hidden: [num_layers * num_directions, batch, hidden_size]
        hidden = torch.zeros(1, batch_size, n_hidden)
        # input_batch: [batch_size, n_step, n_class]
        outputs = model(hidden, input_batch)

        # outputs: [batch_size, n_class], 
        # target_batch: [batch_size] (distributed, not one-hot)
        loss = criterion(outputs, target_batch)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    inputs = [sen.split()[:2] for sen in sentences]

    # Predict
    hidden = torch.zeros(1, batch_size, n_hidden)
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])