# %%
# code changed from Tae Hwan Jung @graykode

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False) # not repetitive

    for i in random_index:  # for sample in one_batch
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # one-hot vec to represent specific central word
        random_labels.append(skip_grams[i][1])  # to predict context (a neighbor in one side)

    return random_inputs, random_labels

class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Linear(voc_size, embedding_size, bias=False)
        self.WT = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        hidden_layer = self.W(X)
        output_layer = self.WT(hidden_layer)
        return output_layer

if __name__ == '__main__':
    batch_size = 2  # mini-batch size
    embedding_size = 2  # embedding size

    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]
    
    word_sequence = ' '.join(sentences).split()
    word_list = ' '.join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    #  word_sequence: ['apple', 'banana', 'fruit', 'banana', ...]
    #  say word_dict: {'apple': 1, 'banana': 2, 'fruit': 3, ...}
    # then skip_gram: [[2, 1], [2, 3], [3, 2], [3, 2], ...]
    skip_grams = []
    for i in range(1, len(word_sequence) - 1): # avoid void in both ends
        # window size only set for 1
        targets = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

        for w in context:
            skip_grams.append([targets, w]) # 2 target-context pairs
    
    # follows are templates
    model = Word2Vec()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch)

        # output: [batch_size, voc_size], target_batch: [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.6f}'.format(loss))
    
    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        # W: [embed_size, voc_size]   ([2, 8])
        x, y = W[0][i].item(), W[1][i].item() # since it gets only 2 embedding size
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()