import numpy as np
from collections import defaultdict
import gridworld_render as render_helper

class GridWorld():
    def __init__(self):
        self.reward_map = np.array([[0,0,0,1],
                               [0,None,0,-1],
                               [0,0,0,0],])
        
        self.action_space = [0,1,2,3]
        self.action_name = ['up','down','left','right']
        self.moving_state = {0: [-1,0], 1: [1,0], 2: [0,-1], 3: [0,1]}

        self.goal_state = (0,3)
        self.wall_state = (1,1)
        self.start_state = (2,0)
        self.agent_state = self.start_state

    def states(self):
        for i in range(self.height):
            for j in range(self.width):
                yield (i,j)

    def next_state(self, state, action):
        new_position = np.array(state) + np.array(self.moving_state[action])
        if new_position[0] < 0 or new_position[0] >= self.height or new_position[1] < 0 or new_position[1] >= self.width:
            new_position = state
        elif new_position[0] == self.wall_state[0] and new_position[1] == self.wall_state[1]:
            new_position = state
        return tuple(new_position)
    
    def reward(self, state, action, next_state):
        return self.reward_map[next_state]
    
    @property
    def width(self):
        return self.reward_map.shape[1]
    
    @property
    def height(self):
        return self.reward_map.shape[0]
    
    @property
    def shape(self):
        return self.reward_map.shape
    
    def actions(self):
        return self.action_space

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

        new_V = 0
        for action, action_proab in pi[state].items():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            new_V += action_proab * (reward + gamma * V[next_state])
        V[state] = new_V
    return V    

def policy_eval(pi, V, env, gamma, threshold=0.001):
    iter = 0
    while True:
        iter += 1
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = 0
        for state in env.states():
            delta = max(delta, abs(old_V[state] - V[state]))
        if delta < threshold:
            print(f'Policy evaluation converged at {iter}th iteration')
            break
        
    return V

def print_value(V, env):
    for i in range(env.height):
        for j in range(env.width):
            print(f'{V[(i,j)]:.2f}', end='\t')
        print()

π = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
γ = 0.9

env = GridWorld()
V = defaultdict(lambda: 0)

v = policy_eval(π, V, env, γ)
print_value(V, env)
env.render_v(v=V)