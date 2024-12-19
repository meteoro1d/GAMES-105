import numpy as np
from scipy.spatial.transform import Rotation as R

r1 = R.from_euler('XYZ', [90, 90, 0], degrees=True)

r2x = R.from_euler('XYZ', [90, 0, 0], degrees=True)
r2y = R.from_euler('XYZ', [0, 90, 0], degrees=True)

r2 = r2y * r2x

r3 = r2x * r2y
print(r1.as_matrix())
print(r2.as_matrix())
print(r3.as_matrix())
