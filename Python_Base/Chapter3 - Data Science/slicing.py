import numpy as np

# Data: yearly salary in ($1000) [2025, 2026, 2027]
dataScientist = [130, 132, 137]
productManager = [127, 140, 145]
designer = [118, 118, 127]
softwareEngineer = [129, 131, 137]

# 注意，在创建 np array 时候，这里默认是整数，故下面有什么操作都不会有小数点
# 想要小数点就要显示指明数据类型
employees_int = np.array([dataScientist, productManager, designer, softwareEngineer])
print(employees_int.dtype)
employees = np.array([dataScientist, productManager, designer, softwareEngineer], dtype=np.float16)
print(employees.dtype)

# 隔一年增加员工 10% 的薪水
# 注意，这里乘于后面的 1.1，numpy 会自动 broadcasting 到相同维度
employees[:, ::2] = employees[:, ::2] * 1.1
print(employees)

# 只增加数据科学家的薪水 20%
employees[0, ::2] = employees[0, ::2] * 1.1
print(employees)
