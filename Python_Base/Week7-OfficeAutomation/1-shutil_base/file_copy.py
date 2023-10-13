from shutil import copy, move, make_archive,unpack_archive
import os

# copy('./test1.txt', './dst1.txt')
path = os.path.join(os.getcwd(), 'test1.txt')
target = os.path.join(os.getcwd(), 'abc.txt')

copy(path, target)

move('./abc.txt', 'bca.txt')

# 以下两个函数都要注意参数含义，可以上网搜
# 压缩
make_archive('compressed', 'zip', os.getcwd())
# 解压缩
unpack_archive('compressed.zip', './compressed')