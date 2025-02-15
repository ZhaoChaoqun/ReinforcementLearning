import numpy as np

np.random.seed(0)

Q = 0
for n in range(1, 11):
    reward = np.random.rand()
    Q = Q + (reward - Q) / n
    print(f'{n}\t{Q:.4f}')