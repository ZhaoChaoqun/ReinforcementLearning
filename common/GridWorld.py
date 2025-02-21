import numpy as np
from collections import defaultdict

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.gridworld_render as render_helper

class GridWorld:
    '''
    环境会根据action输出state
    环境会输出reward
    '''
    def __init__(self):
        self.reward_map = np.array([[0,0,0,1],
                                    [0,None,0,-1],
                                    [0,0,0,0]])
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state
        self.action_space = [0, 1, 2, 3]
        self.action_meaning = { 0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.moving_state = { 0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    @property
    def height(self):
        return self.reward_map.shape[0]
    
    @property
    def width(self):
        return self.reward_map.shape[1]

    def actions(self):
        return self.action_space

    def states(self):
        for i in range(self.height):
            for j in range(self.width):
                yield (i, j)

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        next_state = self.__next_state(self.agent_state, action)
        reward = self.__reward(self.agent_state, action, next_state)
        self.agent_state = next_state
        done = next_state == self.goal_state
        return next_state, reward, done

    def __next_state(self, state, action):
        next_state = (state[0] + self.moving_state[action][0], state[1] + self.moving_state[action][1])
        if next_state[0] < 0 or next_state[0] >= self.height or next_state[1] < 0 or next_state[1] >= self.width:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state
        return next_state

    def __reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)
    
    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)

def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        G = 0
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            G += pi[state][action] * (r + gamma * V[next_state])

        V[state] = G
    return V

def policy_eval(pi, V, env, gamma=0.9, episilon=0.001):
    iter = 0
    while True:
        iter += 1
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = 0
        for state in env.states():
            delta = max(delta, abs(old_V[state] - V[state]))
        if delta < episilon:
            break
    print(f'Value iter: {iter}th')
    return V

def greedy_policy(V, env, gamma=0.9):
    pi = defaultdict(lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    for state in env.states():
        if state == env.goal_state:
            continue
        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            action_values[action] = r + gamma * V[next_state]
        max_action = max(action_values, key=action_values.get)
        for action in env.actions():
            pi[state][action] = 1 if action == max_action else 0
        
    return pi

def policy_iter(env):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)
    iter = 0
    while True:
        iter += 1
        V = policy_eval(pi, V, env)
        old_pi = pi.copy()
        pi = greedy_policy(V, env)
        if old_pi == pi:
            break
        
    print(f'Policy iter: {iter}th')
    return V, pi

def value_onestep(V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            continue
        action_values = {}
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            action_values[action] = r + gamma * V[next_state]
        max_action = max(action_values, key=action_values.get)
        V[state] = action_values[max_action]
    return V

def value_iter(env, gamma=0.9, episilon=0.001):
    V = defaultdict(lambda: 0)
    iter = 0
    while True:
        iter += 1
        old_V = V.copy()
        V = value_onestep(V, env, gamma)

        delta = 0
        for state in env.states():
            delta = max(delta, abs(old_V[state] - V[state]))
        if delta < episilon:
            break
    print(f'Value iter: {iter}th')
    return V

if __name__ == '__main__':
    env = GridWorld()
    print(f'Height: {env.height}, Width: {env.width}')

    # env.eval_onestep()
    # env.render_v(env.V)    
    # env.eval_iter(pi)
    # env.render_v(env.V)

    # V, pi = value_iter(env)
    # env.render_v(V, pi)

    # pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    # V = defaultdict(lambda: 0)
    # V = value_onestep(V, env)
    # env.render_v(V)

    V= value_iter(env)
    # env.render_v(V)
    pi = greedy_policy(V, env)
    env.render_v(V, pi)
