import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[1], [2], [3]])
c = np.array([1, 2, 3])
print c.reshape(-1, 1).shape
A = np.append(A, c.reshape(-1, 1), axis=1)
print A

a = np.array([1, 2, 3])
b = np.array([3, 3, 3])
print(np.sum(a==b))
