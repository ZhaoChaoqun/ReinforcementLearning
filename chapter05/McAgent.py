import numpy as np
from collections import defaultdict

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.GridWorld import GridWorld

np.random.seed(42)


class McAgent():
    def __init__(self):
        self.gamma = 0.9
        self.episilon = 0.1
        self.alpha = 0.1
        self.action_size = 4
        self.base_proabs = {action: self.episilon / self.action_size for action in range(self.action_size)}
        self.Q = defaultdict(lambda: 0)
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

    def greedy_policy(self, state):
        qs = [self.Q[state, action] for action in range(self.action_size)]
        max_action = np.argmax(qs)
        self.pi[state] = self.base_proabs.copy()
        self.pi[state][max_action] += 1 - self.episilon 

    def update(self):
        G = 0
        for state, action, reward in reversed(self.memory):
            G = reward + self.gamma * G
            self.cnts[(state, action)] += 1
            # self.Q[(state, action)] += (G - self.Q[(state, action)]) / self.cnts[(state, action)]
            self.Q[(state, action)] += (G - self.Q[(state, action)]) * self.alpha
            self.greedy_policy(state)

def eval_onestep(agent, env):
    state = env.reset()
    agent.reset()
    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.add(state, action, reward)
        if done:
            agent.update()
            break
        state = next_state

def policy_evaluation(agent, env):
    for _ in range(10000):
        eval_onestep(agent, env)


env = GridWorld()
agent = McAgent()

policy_evaluation(agent, env)
env.render_q(agent.Q)