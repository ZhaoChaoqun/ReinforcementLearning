import numpy as np

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.GridWorld import GridWorld

import torch
import torch.nn as nn

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)
    

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.lr = 0.01
        self.action_dim = 4
        self.state_dim = 12

        self.q_net = QNet(self.state_dim, self.action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_net(torch.tensor(state, dtype=torch.float32))
                return torch.argmax(q_values).item()
            
    def update(self, state, action, reward, next_state, done):
        done = int(done)
        next_qs = self.q_net(torch.tensor(next_state, dtype=torch.float32))
        next_q = torch.max(next_qs, axis=1)
        next_q = next_q.values.item()

        target = reward + (1- done) * self.gamma * next_q
        qs = self.q_net(torch.tensor(state, dtype=torch.float32))
        q = qs[:, action]
        loss = self.loss_fn(q, torch.tensor([target], dtype=torch.float32))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def one_hot(state):
    WIDTH, HEIGHT = 4, 3
    y, x = state
    one_hot = np.zeros(WIDTH * HEIGHT)
    one_hot[y * WIDTH + x] = 1
    one_hot = one_hot[np.newaxis, :]
    return one_hot

def train(agent, env):
    episodes = 400
    loss_history = []
    for episode in range(episodes):
        state = env.reset()
        state = one_hot(state)
        total_loss, cnt = 0, 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = one_hot(next_state)
            loss = agent.update(state, action, reward, next_state, done)
            total_loss += loss
            cnt += 1
            state = next_state
        loss_history.append(total_loss / cnt)
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}, Loss: {total_loss / cnt:.4f}")
    print(f"Episode {episode}/{episodes}, Loss: {total_loss / cnt:.4f}")

# import matplotlib.pyplot as plt
# plt.plot(loss_history)
# plt.xlabel('Episode')
# plt.ylabel('Loss')
# plt.title('Training Loss')
# plt.show()


env = GridWorld()
agent = QLearningAgent()
train(agent, env)

Q = {}
for state in env.states():
    qs = agent.q_net(torch.tensor(one_hot(state), dtype=torch.float32)).detach().numpy()
    for action, q in enumerate(qs[0]):
        Q[(state, action)] = q
env.render_q(Q)

    
