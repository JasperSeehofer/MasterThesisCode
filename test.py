import numpy as np

xs = np.arange(-100, 100)
ys = [1,2,3,4,5,6,7,8,9,0,134,2345,345,12]

indices = np.round(np.linspace(0, len(xs) - 1, 10)).astype(int)

print(xs[indices], ys[indices])