# %%
# code changed from Tae Hwan Jung @graykode

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = nn.Embedding(vocab_size, embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        # embed => conv2d => activ => pool => concat => fcn
        embedded_chars = self.W(X) # [batch_size, sequence_len, sequence_len]
        embedded_chars = embedded_chars.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_len, embedding_size]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            # conv: [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(embedded_chars))
            # mp: ((filter_height, filter_width))
            mp = nn.MaxPool2d((sequence_len - filter_sizes[i] + 1, 1))
            # pooled: [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        # h_pool: [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool = torch.cat(pooled_outputs, len(filter_sizes))
        # h_pool_flat: [batch_size(=6), output_height * output_width * (output_channel * 3)]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])
        # model: [batch_size, num_classes]
        model = self.Weight(h_pool_flat) + self.Bias
        return model

if __name__ == '__main__':
    embedding_size = 2
    sequence_len = 3
    num_classes = 2
    filter_sizes = [2, 2, 2] # n-gram windows
    num_filters = 3

    # 3 words sentences => sequence_len = 3
    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0] # 1 for positive, 0 for negative

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = TextCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # the way to set `inputs` is awesome
    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    targets = torch.LongTensor([out for out in labels]) # To using Torch Softmax Loss

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        outputs = model(inputs)

        # output: [batch_size, num_classes], target_batch: [batch_size] (distributed, not one-hot)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    # Test
    test_text = 'sorry hate you'
    tests = [np.asarray([word_dict[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests)

    # Predict
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Negative meaning...")
    else:
        print(test_text, "is Positive meaning!!!")