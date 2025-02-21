import numpy as np
from collections import defaultdict

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.GridWorld import GridWorld

np.random.seed(42)


class RandomAgent():
    def __init__(self):
        self.gamma = 0.9
        self.V = defaultdict(lambda: 0)
        self.pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        self.memory = []
        self.cnts = defaultdict(lambda: 0)

    def reset(self):
        self.memory.clear()

    def get_action(self, state):
        action_proabs = self.pi[state]
        action = np.random.choice(list(action_proabs.keys()), p=list(action_proabs.values()))
        return action

    def add(self, state, action, reward):
        self.memory.append((state, action, reward))

    def eval(self):
        G = 0
        for state, action, reward in reversed(self.memory):
            G = reward + self.gamma * G
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]


def eval_onestep(agent, env):
    state = env.reset()
    agent.reset()
    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.add(state, action, reward)
        if done:
            agent.eval()
            break
        state = next_state

def policy_evaluation(agent, env):
    for _ in range(1000):
        eval_onestep(agent, env)

def policy_evaluation2(agent, env):
    iter = 0
    for _ in range(1000):
        iter += 1
        old_V = agent.V.copy()
        eval_onestep(agent, env)
        delta = 0
        for state in env.states():
            delta = max(delta, abs(old_V[state] - agent.V[state]))
        if delta < 0.001:
            print(f'Value iter: {iter}th')
            break

env = GridWorld()
agent = RandomAgent()

policy_evaluation2(agent, env)
env.render_v(agent.V)