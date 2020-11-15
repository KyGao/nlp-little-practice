# Skip-Gram

训练词向量有很多方法：Skip-gram, CBOW, GloVe, RNN/LSTM, ...其中 Skip-gram 做的事情就是通过中间单词预测两边的意思，表现在把单词的几何距离相近的词再拉近一些。

代码实现上其实非常简洁，实际上就是三层的全连接网络，所以模型只有俩参数：输入矩阵 $W$，输出矩阵 $W^T$。区别Skip-Gram的特点就是 input 是中心词，label 是某一侧（左/右）的邻居词。数据形式上，input: {one-hot vec} for central word. label: {int} for context word。

可视化的时候提取的是输入矩阵来画，即输入层到隐藏层的权值矩阵。
