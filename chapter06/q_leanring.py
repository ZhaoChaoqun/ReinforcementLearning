import numpy as np
from collections import defaultdict, deque

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.GridWorld import GridWorld

np.random.seed(42)

class QLearningAgent():
    def __init__(self):
        self.Q = defaultdict(lambda: 0)
        self.episilon = 0.1
        self.action_size = 4

        self.gamma = 0.9
        self.alpha = 0.8
        
    def get_action(self, state):
        if np.random.rand() < self.episilon:
            return np.random.choice(self.action_size)
        qs = [self.Q[state, action] for action in range(self.action_size)]
        return np.argmax(qs)
    
    def update(self, state, action, reward, next_state):
        next_qs = [self.Q[next_state, action] for action in range(self.action_size)]
        target_value = reward + self.gamma * max(next_qs)
        self.Q[(state, action)] += self.alpha * (target_value - self.Q[(state, action)])

def update_onestep(agent, env, state):
    action = agent.get_action(state)
    next_state, reward, done = env.step(action)
    agent.update(state, action, reward, next_state)
    return next_state, done

def value_update_iter(agent, env):
    for _ in range(1000):
        state = env.reset()
        done = False
        while not done:
            state, done = update_onestep(agent, env, state)

env = GridWorld()
agent = QLearningAgent()

value_update_iter(agent, env)
env.render_q(agent.Q)
