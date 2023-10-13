lst_1 = [1, 2, 3]
lst_2 = [4, 5, 6]

zipped = list(zip(lst_1, lst_2))
print(zipped)

# *符号作用是去掉数组外面的 []
unzipped = list(zip(*zipped))
print(unzipped)

column_names = ['name', 'salary', 'job']
db_rows = [('Alice', 180000, 'data scientist'),
           ('Bob', 99000, 'mid-level manager'),
           ('Frank', 87000, 'CEO')]

# 将每个数据都和 column_names 结合
db = [dict(zip(column_names, row)) for row in db_rows]
print(db)
