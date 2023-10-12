import torch
from matplotlib import pyplot as plt
import torch.optim as optim


def model(t_u, w, b):
    return w * t_u + b


# 差分张量
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


w = torch.ones(())
b = torch.zeros(())

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
t_p = model(t_u, w, b)
print(t_p)

loss = loss_fn(t_p, t_c)
print(loss)

# 梯度下降减小损失
delta = 0.1

# 模拟导数定义
loss_rate_of_change_w = (loss_fn(model(t_u, w + delta, b), t_c) - loss_fn(model(t_u, w - delta, b), t_c)) / (
        2.0 * delta)
learning_rate = 1e-2
w = w - learning_rate * loss_rate_of_change_w

# b 的改变也一样
loss_rate_of_change_b = (loss_fn(model(t_u, w, b + delta), t_c) - loss_fn(model(t_u, w, b - delta), t_c)) / (
        2.0 * delta)
learning_rate = 1e-2
b = b - learning_rate * loss_rate_of_change_b


# 梯度
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


def dmodel_dw(t_u, w, b):
    return t_u


def dmodel_db(t_u, w, b):
    return 1.0


# 放在一起就是 w和b 的损失梯度函数
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)

    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


def training_loop(n_epochs, learning_rate, params, t_u, t_c, print_params=True):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)
        params = params - learning_rate * grad
        if print_params:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params


t_un = 0.1 * t_u


# params = training_loop(
#     n_epochs=5000,
#     learning_rate=1e-2,
#     params=torch.tensor([1.0, 0.0]),
#     t_u=t_un,
#     t_c=t_c,
#     print_params=False
# )
#
# t_p = model(t_un, *params)

# fig = plt.figure()
# plt.xlabel('Temperature (°Fahrenheit)')
# plt.ylabel('Temperature (°Celsius)')
# plt.plot(t_u.numpy(), t_p.detach().numpy())
# plt.plot(t_u.numpy(), t_c.numpy(), 'o')
# plt.show()


# pytorch 的自动求导
def training_loop_pytorch(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:
            params.grad.zero_()

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        # 反向传播自动计算导数（.grad）
        loss.backward()

        with torch.no_grad():
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

    return params


# training_loop_pytorch(
#     n_epochs=5000,
#     learning_rate=1e-2,
#     params=torch.tensor([1.0, 0.0], requires_grad=True),
#     t_u=t_un,
#     t_c=t_c
# )

learning_rate = 1e-5
params = torch.tensor([1.0, 0.0], requires_grad=True)
optimizer = optim.SGD([params], lr=learning_rate)
print(params)

t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c)
loss.backward()
optimizer.step()
print(params)

# 其它优化器
learning_rate = 1e-1
params = torch.tensor([1.0, 0.0], requires_grad=True)
optimizer = optim.Adam([params], lr=learning_rate)

# training_loop_pytorch(
#     n_epochs=2000,
#     optimizer=optimizer,
#     params=params,
#     t_u=t_u,
#     t_c=t_c
# )

# 打乱数据集
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

# randperm 就能洗乱
shuffled_indices = torch.randperm(n_samples)

# 分割训练和测试数据集
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]
print(train_indices)
print(val_indices)
