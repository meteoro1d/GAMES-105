import numpy as np

a = np.random.random([2, 3])
b = np.random.random([2, 3])
c = np.concatenate((a, b))

print(a)
print('\n')

print(b)
print('\n')

print(c)
print('\n')
print(c.shape)

d = [0, 1, 2, 3, 4, 4]


