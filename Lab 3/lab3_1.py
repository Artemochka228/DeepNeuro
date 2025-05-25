# -*- coding: utf-8 -*-
"""
Created on Sun May 25 15:34:32 2025

@author: artem
"""

import torch 
import numpy as np

x = torch.randint(1, 11, (1, ), dtype=torch.int32)
print(x)

x = x.to(dtype=torch.float32)
print(x)

x.requires_grad=True

y = x ** 3
z = y * torch.randint(1, 11, ())
out = z.exp()
print(out)

out.backward()
print(x.grad)


