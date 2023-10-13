# 注意这不是一行，这是用逗号分割了的 3 个字符串元素
txt = ['lambda functions are anonymous functions.',
       'anonymous functions dont have a name.',
       'functions are objects in Python.']

mark = map(lambda s: (True, s) if 'anonymous' in s else (False, s), txt)
print(list(mark))
