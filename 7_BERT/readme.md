# BERT

建议按这个顺序：`BERT`（核心之核心）=> `Embedding` => `make_batch` => `EncodeLayer, Multihead, Poswise...` 一系列Transformer照搬过来的东西 => `BERT`

### Embedding

总共三种 embedding：`tok_embed`, `pos_embed`, `seg_embed`。三个 embed 的输出维度都是 `d_model`，表示 hidden_size，常见值是 `768`。，所以为一区别就在输入上面，`tok_embed` 的输入就是整个 inputs，形式是每个词/token用一个number表示。至于另外俩输入，虽说是变量，但是一般情况下都是超参，比如其中 `seg_embed` 的 `n_segments=2`， `pos_embed` 中的 `maxlen=30`。

### make_batch

这里的 `make_batch` 可是最复杂的部分。首先在构建 batch 之前还需要对输入进行预处理，需要给这四个 token 留一个具体的位置：`'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3`，所以其他单词的编码都是顺延 4 位（`i + 4`）。

然后开始讲 `make_batch`：

1. 首先就出现一个 `negative & positive` 的奇怪东西，在这里是用作采样过程的控制。希望每个batch里包含（参考 **Contrastive Learning**）相同比例的正负例，这个的正负例在这里是“b是否为a的下一句话”。所以简单来说就希望每个batch里，连续的句子和不连续的句子各一半。
2. 之后对于采样ok的样本就来构造输入token。`input_ids` 就是两句话拼起来然后加上一些 special token 的样子，还包括 `segment_ids` 的初始化（毕竟句子长度也都知道了）。
3. Masked Language Model。`0.15` 表示序列长度的 15% 要 mask 掉，然后要 mask 掉的那些词，`0.8` 表示其中 80% 替换为正常的 `[MASK]` token，`0.5` 表示 10% （剩下的50%） 要拿来随便变成另一个单词，最后 10% 不处理。对以上就是 Mask 的处理。
4. 最后再来 `zero_padding` 的过程，就是把句子补长。然后就是跟 negative 和 positive 的数量更新，应该看得懂。

### EncoderLayer

参考隔壁 Transformer 的解释：

### BERT

前面的都讲了还比较好理解，这里就是单独讲一下两个 loss 是怎么回事。

1. `logits_clsf`。是一个二维的向量，表示模型预测出来两句话有没有上下文关系的一个 logit，之后和 ground truth 就是 `isNext` 送去 CrossEntropy 算出来具体的分类loss。此外讲一下这里的 `h_pooled`为什么这么操作，具体的维度我觉得还是放出来清晰一点：

   ```python
   # batch_size=6, maxlen=30, d_model=768
   # print(output.shape)             : torch.Size([6, 30, 768])
   # print(enc_self_attn.shape)      : torch.Size([6, 12, 30, 30])
   # print(enc_self_attn_mask.shape) : torch.Size([6, 30, 30])
   ```

   `[CLS]` 在第一个位置判断两句话有没有上下文预测，所以取位置0的元素出来，过一遍 FNN 送给激活函数+FNN到2维，再送去criterion。

2. `logits_lm`。里面的 `torch.gather` 可真难，目的是对齐 mask 的顺序！！（ground_truth 与 预测结果）比如 `masked_pos`（ground_truth）就是列出了哪些地方有mask，这里是三维的数据，但是其中一维都是简单的expand（`[1,1,1,1]`）这样，所以这里压缩这个没用的维来举例：比如`[3, 5, 13, 0, 0]`是mask掉的位置。然而就是 `masked_pos` 把mask揉到一起，但是一起做 CrossEntropy 的可是 `output`，顺序是对不上的，所以需要调整一下顺序，就是靠的 `torch.gather`。注意这里 `masked_pos` 和 `masked_token` 是不一样的，前者标识位置信息，用来gather预测输出，后者用来和调整顺序后的预测输出做 CE Loss（具体内容就是每个token的idx，是int）。