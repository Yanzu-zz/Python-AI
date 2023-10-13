text = '''
Call me Ishmael. Some years ago - never mind how long precisely - having
little or no money in my purse, and nothing particular to interest me
on shore, I thought I would sail about a little and see the watery part
of the world. It is a way I have of driving off the spleen, and regulating
the circulation. - Moby Dick'''

# filtering the words which length is greater than three
# 这样写有 \n 问题
# 故我们可以先将 \n 分割，然后再分割这一个个数组
out1 = [word for word in text.split(' ') if len(word) > 3]
print(out1)

out2 = [[word for word in line.split(' ') if len(word) > 3] for line in text.split('\n')]
print(out2)

# 我们将他合成一个数组
out3 = [word for line in text.split('\n') for word in line.split(' ') if len(word) > 3]
print(out3)
