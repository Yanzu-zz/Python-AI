import numpy as np
import matplotlib.pylab as plt


# 计算导数（用定义，这里稍微数学变换了下）
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


# 绘制简单函数图像
# 以 0.1 为步长生成从 1-20 的数组
x = np.arange(0.0, 20, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)


# plt.show()

# 计算梯度
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # x+h
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # x-h
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 加入最终数组中
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


# lr = learning rate 学习率
# step_num 是梯度法的重复次数（也就是逼近极值点几次）
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
