How to make hidden shape 是个难题，为什么是(1, batch_size, hideen)

What differs from output and target: output 是以 **'S' 开头**，target 是以 **'E‘ 结尾**（注意这俩指的是 token）。同时 output 是 **one-hot** 编码，而 target 就是每个字母代表一个idx的 **idx list**。因此后面就是拿decoder输出的list拿来算交叉熵。此外没有别的区别了，单词内容一模一样。

在 `seq2seq_attn.py` 中，data 是有三维的就是这个道理。一维输入是原文，另外两维的区别就是一个开头是 S，一个结尾是 E。`seq2seq.py` 中就是把这俩看成一样，在处理过程中添加 token。这两种处理方式都可行，只要保证在 `make_batch` 之后是对的就行。