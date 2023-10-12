import torch

a = torch.rand(2, 3)
b = torch.rand(3)
print(a)
print(b)


# 此时 b自动扩维成-> [1,3]
# 那么 c.shape = [2,3]
c = a + b
print(c)
print(c.shape)


a1 = torch.rand(2,1,1,3)
b1 = torch.rand(4,2,3)
# c.shape = 2*4*2*3
c1 = a1+b1
print(a1)
print(b1)
print(c1)
