# Transformer

又发现一个宝藏博客：[how-to-code-the-transformer-in-pytorch](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#1b3f)

> **Note: imgaes are all from above blog**

本身结构足够复杂了...可以按这个顺序来看代码：
`EncoderLayer` => `MultiHeadAttention` => `ScaledDotProductAttention` => `PoswiseFeedForwardNet` => `DecoderLayer` => `Encoder` => `Decoder`

### EncoderLayer

众所周知，EncoderLayer 有两个部分: `Self-Attn => FFN`，单纯的两个 block 级联，所以重点在两个 block 内部。

### MultiHeadAttention

MultiHead 中 `attn_mask` 的作用：

> **In the encoder and decoder**: To zero attention outputs wherever there is just padding in the input sentences.
> 
> **In the decoder**: To prevent the decoder ‘peaking’ ahead at the rest of the translated sentence when predicting the next word.

所以对于所有 Head 来说这个 mask 应该都一样，因为都是对于同一个输入。具体怎么构造这个 mask 在后面讲 Encoder 的时候会说。

有一个非常憨憨的错觉我还是放在这里小心下次又想歪了...虽然说是 $q_s = W^Q \cdot x_s$，但是代码实现并不是初始化这个 $W^Q$ 矩阵！！（没错我这里还找了一段时间并且对 `nn.Linear()` 疑惑不解）而是用一个 `nn.Linear(dim_of_q_s, dim_of_x_s)` 这样的 block 来包含了这个 $W^Q$，一般人应该不会犯我这种错...MultiHead Attention 实现建议参考下图：

![multihead](https://miro.medium.com/max/627/1*1tsRtfaY9z6HxmERYhw8XQ.png)

### ScaledDotProductAttention

这里在做的事刚好也可以用下图来概括：

![dotproduct](https://miro.medium.com/max/168/1*15E9qKg9bKnWdSRWCyY2iA.png)

### PoswiseFeedForwardNet

可以看到是正常的三层神经网络

注意这里和 `MultiHeadAttention` 的共性是输出时都有 residual 处理和 LayerNorm，也就是说二者在 return 的时候是一样的：`self.layer_norm(output + residual)`

### DecoderLayer

众所周知，DecoderLayer有三块：`Self-Attn => EncDec-Aten => FFN`。其中 `Self-Attn` 和 `FFN` 与 EncoderLayer 的完全一样，唯一区别就是 mask 的东西不同，那个还不是现在看的内容，所以先过。另外这里 `EncDec-Attn` 的对象不一样，虽然也是调用 MultiHeadAttention 模块，但是是希望算出 `Q = dec_outputs, K = enc_outputs` 之间的 Attention。需要注意的是这里 `V` 取的是 `enc_outputs` 而不是 `dec_outputs`，就像 `Seq2Seq_Attn.py` 中也是得到了 attntion 去和 `enc_outputs` 做 batch matrix-matrix product。

### Encoder

实现整体思路应该问题不大，但是要掌握的细节还蛮多的：

1. EncoderLayer 级联的方式：熟悉的 `nn.ModuleList()`，然后在 `forward()`函数中用 `for layer in self.layers` 巧妙调用
2. 词的 Embedding 表示：词向量 + 位置向量。词向量不用讲，位置向量emmm，太过于复杂而且与其他知识耦合程度小所以有机会回来学...可以提一句这里位置向量的编码是借助 `from_pretrained` 得到的，暂时不知道优势在哪2333
3. **Mask的设置**：前面有说到为什么要 mask，在 Encoder 中，需要把 padding 补上来的部分 mask 掉。刚好 padding 的部分值为0, 所以 mask 只需要找值 `.data.eq(0)` 的就自动生成 mask 矩阵了

### Decoder

和 Encoder 的区别就在于 mask 的设置不同。

前面有说到为什么要 mask，在 Decoder 中，`EncDec-Attn` 是只需要照顾这个 padding mask，因为主要过的是 encoder，不会看到 decoder 的未来信息。而 `Self-Attn` 就不一样了，为了不看到未来信息，具体方法是直接取这个 `dec_inputs` 的上三角矩阵，此外与 padding 的 mask 求并集作为最终的 mask。

### Transformer

这不就来了，其实看到这里就很简单了：就是 Encoder 接 Decoder 再做个 Linear 映射到 logits，最后分类啥的就是正常的 CELoss 了。