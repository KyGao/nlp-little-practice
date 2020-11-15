# TextRNN

都不需要啥参考文献，就是最朴素的 RNN for classification。结构就是 `RNN - FCN`。在代码中，就是拿 RNN 最后一层的 output 出来送给 FCN，FCN 的输出就是分类结果。

另外 LSTM 就也是一模一样套api就行

疑问：为什么 hidden 的初始化是全0,甚至在 inference 的时候也是设为全0,hidden 不是应该也是模型参数吗？
