import numpy as np
import matplotlib.pyplot as plt

sequences = np.random.normal(10.0, 1.0, 500)
# print(sequences)

# plt.xkcd()
# plt.hist(sequences)
# plt.annotate(r"$\omega_1=9$", (9, 70))
# plt.annotate(r"$\omega_2=11$", (11, 70))
# plt.annotate(r"$\mu=10$", (10, 90))
# plt.savefig("plot.jpg")
# plt.show()

# Website analytics data:
# (row = day), (col = users, bounce, duration)
a = np.array([[815, 70, 115],
              [767, 80, 50],
              [912, 74, 77],
              [554, 88, 70],
              [1008, 65, 128]])
mean, std = np.mean(a, axis=0), np.std(a, axis=0)
print(mean)
print(std)

print(a[:, 0] - mean[0])
print(np.abs(a[:, 0] - mean[0]))
print(np.abs(a[:, 0] - mean[0]) > std[0])
print(np.abs(a[:, 1] - mean[1]) > std[1])
print(np.abs(a[:, 2] - mean[2]) > std[2])

outliers = (
        (np.abs(a[:, 0] - mean[0]) > std[0]) *
        (np.abs(a[:, 1] - mean[1]) > std[1]) *
        (np.abs(a[:, 2] - mean[2]) > std[2])
)

print()
print(outliers)
