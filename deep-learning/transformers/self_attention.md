

## **Transformer Encoder**

transformer最核心的操作就是self-attention，其实attention机制很早就在NLP和CV领域应用了，比如带有attention机制的[seq2seq模型](https://www.zhihu.com/search?q=seq2seq模型&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})，但是transformer完全摒弃RNN或LSTM结构，直接采用attention机制反而取得了更好的效果：attention is all you need！简单来说，attention就是根据当前查询对输入信息赋予不同的权重来聚合信息，从操作上看就是一种“加权平均”。attention中共有3个概念：query, key 和 value，其中 key 和 value 是成对的，对于一个给定的query向量![[公式]](https://www.zhihu.com/equation?tex=q%5Cin+%5Cmathbb%7BR%7D%5E%7Bd%7D)，通过内积计算来匹配k个[key向量](https://www.zhihu.com/search?q=key向量&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})（维度也是d，堆积起来即[矩阵](https://www.zhihu.com/search?q=矩阵&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})![[公式]](https://www.zhihu.com/equation?tex=K%5Cin+%5Cmathbb%7BR%7D%5E%7Bk%5Ctimes+d%7D)），得到的内积通过softmax来归一化得到k个权重，那么对于query其attention的输出就是k个key向量对应的value向量（即矩阵![[公式]](https://www.zhihu.com/equation?tex=V%5Cin+%5Cmathbb%7BR%7D%5E%7Bk%5Ctimes+d%7D)）的[加权平均值](https://www.zhihu.com/search?q=加权平均值&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})。对于一系列的N个query（即矩阵![[公式]](https://www.zhihu.com/equation?tex=Q%5Cin+%5Cmathbb%7BR%7D%5E%7BN%5Ctimes+d%7D))，可以通过矩阵计算它们的attention输出：

![[公式]](https://www.zhihu.com/equation?tex=Attention%28Q%2C+K%2C+V%29+%3D+Softmax%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%29V+%5C%5C)

这里的![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7Bd_k%7D)为缩放因子以避免[点积](https://www.zhihu.com/search?q=点积&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})带来的方差影响。上述的Attention机制称为**Scaled dot product attention**，其实attention机制的变种有很多，但基本原理是相似的。如果![[公式]](https://www.zhihu.com/equation?tex=Q%2CK%2CV)都是从一个包含![[公式]](https://www.zhihu.com/equation?tex=N)个向量的sequence（![[公式]](https://www.zhihu.com/equation?tex=X%5Cin+%5Cmathbb%7BR%7D%5E%7BN%5Ctimes+D%7D)）通过线性变换得到：![[公式]](https://www.zhihu.com/equation?tex=Q%3DXW_Q%2CK%3DXW_K%2CV%3DXW_V)那么此时就变成了**self-attention**，这个时候就有![[公式]](https://www.zhihu.com/equation?tex=N)个（key,value）对，那么![[公式]](https://www.zhihu.com/equation?tex=k%3DN)。self-attention是transformer最核心部分，self-attention其实就是输入向量之间进行相互attention来学习到新特征。


```python
import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    x = torch.rand(32, 196, 768)
    attn = Attention(dim=768)
    out = attn(x)
    print(out.shape)
```

### 数据流程

假设 输入是 B * N *C = 32 * 196 * 768,  head 的个数为 8,

- 24 行,   32 * 196 * 768 
- 25 行,   32 * 196 * 768  ==>  32 * 196 * (768  * 3 )  ==>  32  * 196 * 3 * 8 * 96  ==>  3 *32  * 8 * 196   *96
- 27 行,   q,k ,v 的shape 都是  32  * 8 * 196   *96
- 30 行,   矩阵乘法,  q * k/sqrt(d) :   32  * 8 * 196   * 196
- 31 行,  计算  softmax,   shape : 32  * 8 * 196   * 196
- 34 行,   计算  softmax(qk/d) * v ,  shape:   32  * 8 * 196   *  96  ==>  32  * 196  * 8 *  96 ==>   32  * 196 * 768

这样， 对于 B * N *C = 32 * 196 * 768 大小的输入，  送入self-attention就能到同样 size 的 sequence 输出，只不过特征改变了。

