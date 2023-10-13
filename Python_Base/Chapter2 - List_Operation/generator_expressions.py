## Data
companies = {
    'CoolCompany': {'Alice': 33, 'Bob': 28, 'Frank': 29},
    'CheapCompany': {'Ann': 4, 'Lee': 9, 'Chrisi': 7},
    'SosoCompany': {'Esther': 38, 'Cole': 8, 'Paris': 18}}

# 只要有一个员工工资 < 最低时薪，就加进结果数组
illegal = [x for x in companies if any(y < 9 for y in companies[x].values())]
print(illegal)
