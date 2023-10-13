import numpy as np

# Data: SAT scores for different students
sat_scores = np.array([1100, 1256, 1543, 1043, 989, 1412, 1343])
students = np.array(["John", "Bob", "Alice", "Joe", "Jane", "Frank", "Carl"])

print(((sorted(zip(students, sat_scores), key=lambda x: -x[1]))[:3]))
tmp = (sorted(zip(students, sat_scores), key=lambda x: -x[1]))[:3]
print([tmp[a][0] for a in range(len(tmp))])

sortedStu = students[np.argsort(sat_scores)]
print(sortedStu)
print(sortedStu[:-4:-1])
