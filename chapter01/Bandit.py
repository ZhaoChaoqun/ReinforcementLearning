import numpy as np

# np.random.seed(0)

class Bandit():
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = np.random.rand()
        if rate > self.rates[arm]:
            return 1
        return 0

bandit = Bandit()
for i in range(3):
    print(bandit.play(0))


ns = np.zeros(10)
Qs = np.zeros(10)

for i in range(10):
    action = np.random.randint(10)

    ns[action] += 1
    Qs[action] += (bandit.play(action) - Qs[action]) / ns[action]
    print(Qs)


class Agent():
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.ns = np.zeros(action_size)
        self.Qs = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.Qs))
        return np.argmax(self.Qs)