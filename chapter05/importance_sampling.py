import numpy as np


np.random.seed(42)
x = np.array([1, 2, 3])
pi = np.array([0.1, 0.1, 0.8])

print(f'Exp {np.sum(x * pi):.4f}')

def random_sample(x, pi):
    samples = []
    for _ in range(1000):
        samples.append(np.random.choice(x, p=pi))
    
    return np.mean(samples), np.var(samples)
mean, var = random_sample(x, pi)
print(f'random_sample mean: {mean:.4f}, var: {var:.4f}')

bi = np.array([1/3,1/3,1/3])

def importance_sampling(x, pi, bi):
    samples = []
    for _ in range(1000):
        idx = np.arange(len(x))
        index = np.random.choice(idx, p=bi)
        samples.append(x[index] * pi[index] / bi[index])
    return np.mean(samples), np.var(samples)
mean, var = importance_sampling(x, pi, bi)
print(f'importance_sampling mean: {mean:.4f}, var: {var:.4f}')

bi = np.array([0.2,0.2,0.6])

mean, var = importance_sampling(x, pi, bi)
print(f'importance_sampling mean: {mean:.4f}, var: {var:.4f}')