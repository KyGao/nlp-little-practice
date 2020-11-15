# TextCNN

论文原文: 

1. [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) 是 TextCNN 开山之作

2. [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820.pdf) 关于如何设计卷积核的深入研究

关于 TextCNN 是什么的问题，其实这张图可以大致说明清楚了（分类模型），像我一样没看懂的可以继续看后面的再回来看这张图，就会觉得 TextCNN 原来如此。

论文1的图：
![TextCNN](https://pic4.zhimg.com/80/v2-6693163cc6f1442e41d82df495868792_720w.jpg?source=1940ef5c)
论文2的图：
![TextCNN2](https://pic3.zhimg.com/v2-2ea1f0b8b166f31273b26bca3ba8e8b2_b.jpg)

## 现存意义

TextCNN 在 NLP 领域是一种效果很好，训练速度很快，且容易实现的网络结构，所以也被各种封装到框架中，即开即用，一般算法比赛都会拿来试一试，对于**短句子文本**很有用。

[知乎上](https://zhuanlan.zhihu.com/p/102366062)有说到 Bi-GRU+conv+pooling 与之相比训练速度慢很多，效果也不占优，尚未考证。

## 结构解析

先解读一下上面的结构图可能对于之后理解原理会更有帮助。其实从代码来看很简单，就是 `embed => (conv2d => activ => pool) => concat => fcn` 结构。其中 conv2d 是由很多个filter组成，希望能提取出不同意思，代码思路就是弄个 Conv2d 的 List（包装在 `nn.ModuleList` 中），然后都正常进行上述括号内的运算，最后把这些 pooling 之后的输出拼接起来就好。

核心是搞懂filter吃进去的是啥，吐出来的是啥：

```python
nn.Conv2d(1, num_filters, (size, embedding_size))
```

可以看到这里输入是 `1` 维，输出是 `num_filter` 维。在CV中，`Conv2d` 输入的tensor长这样: `[batch_size, channel_size, height, width]`。与之不同，TextCNN 的batch_size 表示几段词向量矩阵（n个词向量stack），channel_size 为1（只能为1），关于一张图的描述（h和w）就变成了词向量矩阵（`[seq_len, emb_size]`）。看到 filter 这里，filter 的维度是 `[size, embedding_size]`，相当于就是从这个 sequence 中抽取某几行（大小由size决定），这些行送去做一个联合分布映射，如第一个图中若取 `size=2` 则如红色部分所示，`size=3` 则如黄色部分所示。

在第二篇论文（即TextCNN的后续研究）中，又将结构泛化到更灵活的 filter 配置。默认的 TextCNN 超参参考如下图：

![superparam](https://pic1.zhimg.com/v2-a0de86fee7c073e95ee325fea3ba21f8_b.jpg)

## 原理篇

[小占同学的回答](https://www.zhihu.com/search?type=content&q=TextCNN)说的很好。

卷积神经网络的核心思想是**捕捉局部特征**，对于文本来说，局部特征就是**由若干单词组成的滑动窗口**，类似于N-gram。卷积神经网络的优势在于能够**自动地对N-gram特征进行组合和筛选，获得不同抽象层次的语义信息。**

重新回来看CV中的卷积核，在CV中，卷积核往往都是正方形的，比如 3*3 的卷积核，然后卷积核在整张image上沿高和宽按步长移动进行卷积操作。与CV中不同的是，**在NLP中输入层的"image"是一个由词向量拼成的词矩阵，且卷积核的宽和该词矩阵的宽相同，该宽度即为词向量大小，且卷积核只会在高度方向移动**。因此，每次卷积核滑动过的位置都是完整的单词，不会将几个单词的一部分"vector"进行卷积，词矩阵的行表示离散的符号（也就是单词），这就保证了word作为语言中最小粒度的合理性（当然，如果研究的粒度是character-level而不是word-level，需要另外的方式处理）。

然后，我们详述这个卷积、池化过程。由于卷积核和word embedding的宽度一致，一个卷积核对于一个sentence，卷积后得到的结果是一个vector，其shape=(sentence_len - filter_window_size + 1, 1)，那么，在经过max-pooling操作后得到的就是一个Scalar。我们会使用多个filter_window_size（原因是，这样不同的kernel可以获取不同范围内词的关系，获得的是纵向的差异信息，即类似于n-gram，也就是在一个句子中不同范围的词出现会带来什么信息。比如可以使用3,4,5个词数分别作为卷积核的大小），每个filter_window_size又有num_filters个卷积核（原因是卷积神经网络学习的是卷积核中的参数，每个filter都有自己的关注点，这样多个卷积核就能学习到多个不同的信息。

## 关于 PyTorch 的小知识

在 [吴海波的回答](https://www.zhihu.com/question/67209417/answer/536180457) 中有
有一个知识点需要先了解一下：filter的定义问题。与图像中的 Conv 层不同的是，TextCNN的Conv存在几个并行的filter，所以这些需要用 list 来创建。但是如果创建方式如下：

```python
self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
```

但是这样创建的其实是list，这些参数并没有被计算到整个模型中，是不会被optimizer识别到其中的参数的。所以应该用如下写法：

```python
self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
```
