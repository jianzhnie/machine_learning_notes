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

## 数据流程

假设 输入是 B * N *C = 32 * 196 * 768,  head 的个数为 8,

- 24 行,   32 * 196 * 768 
- 25 行,   32 * 196 * 768  ==>  32 * 196 * (768  * 3 )  ==>  32  * 196 * 3 * 8 * 96  ==>  3 *32  * 8 * 196   *96
- 27 行,   q,k ,v 的shape 都是  32  * 8 * 196   *96
- 30 行,   矩阵乘法,  q * k/sqrt(d) :   32  * 8 * 196   * 196
- 31 行,  计算  softmax,   shape : 32  * 8 * 196   * 196
- 34 行,   计算  softmax(qk/d) * v ,  shape:   32  * 8 * 196   *  96  ==>  32  * 196  * 8 *  96 ==>   32  * 196 * 768