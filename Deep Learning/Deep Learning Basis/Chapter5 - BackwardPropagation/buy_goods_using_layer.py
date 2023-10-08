from layer_naive import *

# 实现书上的例子
print("First example: buy apples")
apple = 100
apple_num = 2
tax = 1.1

# init layer
# 涉及到两个乘法，故我们初始化两个 MulLayer
# 两个指的是先算苹果价钱，然后和 税率 相乘
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 前向传播
# 先算苹果总价钱
apple_price = mul_apple_layer.forward(apple, apple_num)
# 然后算上税率，就等于最终总价钱
price = mul_tax_layer.forward(apple_price, tax)
# 接着查看最终需要给的钱
# print(price)

# 反向传播
dprice = 1
# 注意此时反向传播的调用顺序要和前向传播相反
# 也就是先从 tax 层开始往后
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
# print(dapple, dapple_num, dtax)

# 下面进行第二个例子，买苹果和橘子
print("Second example: buy apples and oranges")
# 涉及多个加乘法
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# init layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange = AddLayer()
mul_tax_layer = MulLayer()

# 看图操作
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
apple_orange_price = add_apple_orange.forward(apple_price, orange_price)
total_price = mul_tax_layer.forward(apple_orange_price, tax)
print(total_price)

# 反向传播
dprice = 1
dapple_orange_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange.backward(dapple_orange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
print(dapple_num, dapple, dorange, dorange_num, dtax)

# print(dapple, dapple_num)
# print(dorange, dorange_num)



#