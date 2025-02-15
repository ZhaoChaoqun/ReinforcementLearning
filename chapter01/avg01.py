import numpy as np

np.random.seed(0)

rewards = []
for n in range(1, 11):
    reward = np.random.rand()
    rewards.append(reward)
    Q = sum(rewards) / n
    print(f'{n}\t{Q:.4f}')
