# NNLM

论文原文：[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

说明参考：[爱罗月的知乎文章](https://zhuanlan.zhihu.com/p/81392113)

非常荣幸第一次实现的 NLP 模型是 2003 年由 Yoshua Bengio 提出的 NNLM 模型，[爱罗月的知乎文章](https://zhuanlan.zhihu.com/p/81392113)提纲挚领地回顾了一下这篇文章。这里也抽出几个点：

- 模型任务：建立语言模型
- 网络结构：![NNLM](https://pic4.zhimg.com/80/v2-62b7f5d9d363a9bd2ec9aacf55d1d8a3_1440w.jpg)
- 步骤：(1) look up table C; (2) the probability function $g$ over words
- softmax 之前的数学模型：$y = b + Wx + U\tanh(d+Hx)$，注意其实有一个类似 residual 的连接。这里x是把所有词的 embedding 直接 concatenate 起来的。此外 table C 也是一个参数，所以总的要训练的参数很多: $\theta = (b, d, W, U, H, C)$