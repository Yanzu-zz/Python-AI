import torch

# add operation
#  定义两个shape相同的矩阵
a = torch.rand(2, 3)
b = torch.rand(2, 3)
print(a)
print(b)

print(a + b)
print(a.add(b))
print(torch.add(a, b))
# 这里操作名字后面加下划线的函数会改变调用变量的值
print(a.add_(b))
# 输出查看
print(a)

# sub
print(a - b)
print(a.sub(b))
print(torch.sub(a, b))
print(a.sub_(b))

# mul
print(a * b)
print(a.mul(b))
print(torch.mul(a, b))
print(a.mul_(b))

# div
print(a / b)
print(a.div(b))
print(torch.div(a, b))
print(a.div_(b))

# 矩阵运算 matmul
a = torch.ones(2, 1)
b = torch.ones(1, 2)

# 矩阵乘法就是@符号
print(a @ b)
print(a.matmul(b))
print(torch.matmul(a, b))
print(a.mm(b))

# 高纬度 Tensor 的运算
a = torch.ones(1, 2, 3, 4)
b = torch.ones(1, 2, 4, 3)

print(a.matmul(b))

# 指数运算 exp
a = torch.tensor([1, 2], dtype=torch.float32)
print(torch.exp(a))
# 注意有下划线的而就会改变原值
print(torch.exp_(a))
print(a.exp())
print(a.exp_())


# 对数
a = torch.tensor([10, 2], dtype=torch.float32)
print(torch.log(a))
print(a.log())
print(a.log_())


# sqrt 开根


