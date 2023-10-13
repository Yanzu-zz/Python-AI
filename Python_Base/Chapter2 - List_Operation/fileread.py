filename = 'readpy.txt'

# 常规写法
f = open(filename)
out1 = []
for line in f:
    out1.append(line.strip())

print(out1)
f.close()

# one-liner 写法
out2 = [line.strip() for line in open(filename)]
print(out2)
