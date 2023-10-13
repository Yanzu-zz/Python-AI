visitors = ['Firefox', 'corrupted', 'Chrome', 'corrupted',
            'Safari', 'corrupted', 'Safari', 'corrupted',
            'Chrome', 'corrupted', 'Firefox', 'corrupted']

# 将前一个值赋给后一个值（也就是覆盖 corrupted）
# 注意，这里左边是从第二个元素（也就是第一行第一个 corrupted 开始）
# 右边则是正常第一个元素（也就是每个 corrupted 的前一个）
visitors[1::2] = visitors[::2]

print(visitors)
