#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time         : 2020/11/25 19:54
# @Author       : LiLin
# @File         : MunkresTest.py
# @Software     : PyCharm
# @Description  :

import numpy as np
#from sklearn.utils.linear_assignment_ import linear_assignment  使用这个要降低版本 降低scikit-learn版本使用<=0.19
from scipy.optimize import linear_sum_assignment

cost_matrix = np.array([
    [15,40,45],
    [20,60,35],
    [20,40,25]
])

# matches = linear_assignment(cost_matrix)
# print('sklearn API result:\n', matches)
matches2 = linear_sum_assignment(cost_matrix)
print('scipy API result:\n', matches2)

"""Outputs
sklearn API result:
 [[0 1]
  [1 0]
  [2 2]]
scipy API result:
 (array([0, 1, 2], dtype=int64), array([1, 0, 2], dtype=int64))
"""