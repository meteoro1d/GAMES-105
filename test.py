import numpy as np

arr = np.random.choice(np.arange(100, dtype=np.int32), size=(3, 15), replace=False)

arr = np.array([1, 2, 3, 4, 5, 6])
arr = arr.reshape(3, 2)
print(arr)
arr = arr.reshape(-1, )
print(arr)
