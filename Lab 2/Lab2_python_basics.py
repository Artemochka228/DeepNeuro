# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:27:38 2025

@author: ПользовательHP
"""

import time
from random import randint
import math as M
    

values = []

for i in range(0, 10):
    values.append(randint(0, 100))
    
result = 0
    
for i in range(0, 10):
    if (values[i] % 2 == 0):
        result += values[i]
        
print(result)

