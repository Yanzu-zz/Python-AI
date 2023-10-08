import numpy as np


# 直接按照公式版本
# def AND(x1, x2):
#   # 权重以及阈值
#   w1, w2, theta = 0.5, 0.5, 0.7
#   tmp = x1 * w1 + x2 * w2
#
#   # 看看计算后的值是否能超过阈值
#   if tmp <= theta:
#     return 0
#   else:
#     return 1

# 我们将公式变形：θ -> -b，然后换边，就得到这个版本
def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7

  # 计算 x1*w1 + x2*w2（python 简洁写法）
  tmp = np.sum(w * x) + b

  # 此时阈值就到了 0（x*w 大于 偏置值 b，则神经元就会激活
  if tmp <= 0:
    return 0
  else:
    return 1


# 与非门
def NAND(x1, x2):
  # 逻辑与 AND 相反（即 !AND）
  x = np.array([x1, x2])
  w = np.array([-0.5, -0.5])
  b = 0.7

  tmp = np.sum(x * w) + b
  if tmp <= 0:
    return 0
  else:
    return 1


# 或门
def OR(x1, s2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.2

  tmp = np.sum(x * w) + b
  if tmp <= 0:
    return 0
  else:
    return 1


# 异或
def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)

  return y


def step_function(x):
  y = x > 0
  return y.astype(np.int)
