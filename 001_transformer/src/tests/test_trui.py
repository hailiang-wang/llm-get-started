'''
https://docs.pytorch.ac.cn/docs/2.10/generated/torch.triu.html
'''

import sys
import torch

# 使用 triu 函数，进行 Mask 遮盖的原理
a = torch.randn(3, 3)
print(a)

b = torch.triu(a, diagonal=1)
print(b)

# sys.exit()
c = torch.transpose(b, dim0=1, dim1=0)

print(c == torch.tensor([0]).float())
