import numpy as np
from sklearn import tree

# Data: student scores in (math, language, creativity) --> study field
X = np.array([[9, 5, 6, "computer science"],
              [1, 8, 1, "linguistics"],
              [5, 7, 9, "art"]])

decision_tree = tree.DecisionTreeClassifier().fit(X[:, :-1], X[:, -1])

students = [[7, 2, 2], [0, 6, 5], [0, 3, 6]]
print(decision_tree.predict(students))

# Result & puzzle
student_0 = decision_tree.predict([[8, 6, 5]])
print(student_0)
student_1 = decision_tree.predict([[3, 7, 9]])
print(student_1)
