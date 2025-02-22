import numpy as np
from collections import defaultdict, deque

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.GridWorld import GridWorld

np.random.seed(42)

class SarsaAgent():
    def __init__(self):
        self.Q = defaultdict(lambda: 0)
        self.pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        self.episilon = 0.1
        self.action_size = 4
        self.base_proabs = {action: self.episilon / self.action_size for action in range(self.action_size)}
        self.memory = deque(maxlen=2)

        self.gamma = 0.9
        self.alpha = 0.1
        
    def reset(self):
        self.memory.clear()

    def get_action(self, state):
        action_proabs = self.pi[state]
        return np.random.choice(list(action_proabs.keys()), p=list(action_proabs.values()))

    def greedy_policy(self, state):
        qs = [self.Q[state, action] for action in range(self.action_size)]
        max_action = np.argmax(qs)
        self.pi[state] = self.base_proabs.copy()
        self.pi[state][max_action] += 1 - self.episilon     

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return
        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        target_value = reward + self.gamma * self.Q[(next_state, next_action)]
        self.Q[(state, action)] += self.alpha * (target_value - self.Q[(state, action)])
        self.greedy_policy(state)

def update_onestep(agent, env, state):
    action = agent.get_action(state)
    next_state, reward, done = env.step(action)
    agent.update(state, action, reward, done)
    return next_state, done


def value_update_iter(agent, env):
    for _ in range(1000):
        state = env.reset()
        agent.reset()

        done = False
        while not done:
            state, done = update_onestep(agent, env, state)
        update_onestep(agent, env, state)

def value_update_iter2(agent, env):
    for _ in range(1000):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, done)

            if done:
                break
            state = next_state
        agent.update(next_state, None, None, None)
        

env = GridWorld()
agent = SarsaAgent()

value_update_iter(agent, env)
env.render_q(agent.Q)
