# %%
# code changed from Tae Hwan Jung @graykode

import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split() # from a sentence to a list
        inputs = [word_dict[n] for n in word[:-1]]
        targets = word_dict[word[-1]]  # predict: last word

        input_batch.append(inputs)
        target_batch.append(targets)

    return input_batch, target_batch

# Model
class NNLM(nn.Module):
    # note the dimension of these 6 params
    # y = b + Wx + U \tanh( d + Hx )
    # [n_cls] = [n_cls] + [n_cls, n_stp*m] * [n_stp*m] + [n_cls, n_hdn] * [n_hdn + [n_hdn, n_stp*m] * [n_stp*m]]
    def __init__(self): # didn't pass in here, the params are set globally
        super(NNLM, self).__init__()
        # nn.Embedding: to store word embeddings and retrieve them using indices
        #     (size of dict, length of each vector)
        self.C = nn.Embedding(n_class, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        # nn.Parameter: to make sth. trainable (updatable). eg. these torch.ones)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        X = self.C(X) # X: [batch_size, n_stp, n_cls]
        X = X.view(-1, n_step * m) # [batch_size, n_stp*n_cls]
        tanh = torch.tanh(self.d + self.H(X))
        output = self.b + self.W(X) + self.U(tanh)
        return output

if __name__ == '__main__':
    n_step = 2 # num of steps, n-1 in paper
    n_hidden = 2 # num of hidden size, h in paper
    m = 2 # embedding size, m in paper

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split() # ['i', 'like', 'dog', 'i', 'love', 'coffee', 'i', 'hate', 'milk']
    word_list = list(set(word_list))        # ['coffee', 'hate', 'love', 'dog', 'milk', 'i', 'like']

    # every time changes
    word_dict = {w: i for i, w in enumerate(word_list)} # {'coffee': 0, 'hate': 1, 'love': 2, 'dog': 3, 'milk': 4, 'i': 5, 'like': 6}
    number_dict = {i: w for i, w in enumerate(word_list)} # {0: 'coffee', 1: 'hate', 2: 'love', 3: 'dog', 4: 'milk', 5: 'i', 6: 'like'}
    n_class = len(word_dict)

    # follows are templates
    model = NNLM()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)  # why LongTensor?
    target_batch = torch.LongTensor(target_batch)
    # p.s. note the diffs between `torch.tensor` and `torch.Tensor`

    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        outputs = model(input_batch)

        # outputs: [batch_size, n_cls], target_batch: [batch_size]
        loss = criterion(outputs, target_batch)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 1000 == 0:
            # the usage of print... exhaustive!
            print('Epoch', '%04d' % (epoch + 1), 'cost=', '{:.6f}'.format(loss))

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1] # tensor([[1],[2], [3]]) => squeeze
    
    # Test
    print([sen.split()[:2] for sen in sentences], '=>', [number_dict[n.item()] for n in predict.squeeze()])