import gymnasium as gym

env = gym.make("LunarLander-v3")  # 设置可视化模式
# env = gym.make("LunarLander-v3", render_mode="human")  # 设置可视化模式
state, info = env.reset()
'''
在 LunarLander-v3 环境中，state 的 shape 为 8 是因为该环境的状态空间由 8 个变量构成，这些变量用于描述登月器的状态。

具体来说，这 8 个变量通常包含以下内容：

1. x 和 y 坐标：登月器在环境中的位置。
2. x 和 y 速度：登月器在两个方向上的速度。
3. 角度：登月器的旋转角度（以弧度表示）。
4. 角速度：登月器的角速度（旋转速度）。
5. 左右推进器的状态：表示左右推进器的开关状态（0 表示关闭，1 表示开启）。
6. 主推进器的状态：表示主推进器的开关状态。

举个例子：
假设在某一时刻，state 可能是这样的一个向量：

state = [x_pos, y_pos, x_velocity, y_velocity, angle, angular_velocity, left_engine, right_engine]
'''
# for _ in range(100):
#     env.render()
#     action = env.action_space.sample()  # 随机选择动作
#     env.step(action)
# env.close()

# --------------------- 定义 DQN 网络结构 ---------------------
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --------------------- 初始化 DQN 网络 ---------------------
# 获取状态和动作空间的大小
state_size = env.observation_space.shape[0]
print(f'''state_size: {state_size}''')
action_size = env.action_space.n
print(f'''action_size: {action_size}''')

# 初始化 DQN 网络
dqn = DQN(state_size, action_size)

# 定义优化器
optimizer = optim.Adam(dqn.parameters(), lr=0.001)

# --------------------- 定义训练过程 ---------------------
from collections import deque
import random

# 定义一个经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# 超参数
buffer_size = 10000
batch_size = 64
gamma = 0.99  # 折扣因子
epsilon = 0.1  # 探索率
epsilon_min = 0.01
epsilon_decay = 0.995
target_update_frequency = 10  # 每隔多少步更新目标网络

# 初始化回放缓冲区
replay_buffer = ReplayBuffer(buffer_size)

# 初始化目标网络
target_dqn = DQN(state_size, action_size)
target_dqn.load_state_dict(dqn.state_dict())

# 训练函数
def train_dqn(episodes):
    global epsilon  # 声明 epsilon 为全局变量
    for episode in range(episodes):
        #----------------------Step 1 Begin ------------------------------------
        # Step 1: 观察状态和动作
        state, info = env.reset()  # 获取初始状态
        done = False
        total_reward = 0

        while not done:
            # epsilon-greedy 策略选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()  # 随机选择动作
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = torch.argmax(dqn(state_tensor)).item()  # 选择 Q 值最大的动作
        #----------------------Step 1 End ------------------------------------
            # 执行动作并获得新状态、奖励
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # 存储经验
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state

            # 进行 Q-learning 更新
            if replay_buffer.size() >= batch_size:
                # 从缓冲区中随机采样一个批次
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.bool)
                #----------------------Step 2 Begin ------------------------------------
                # Step 2: 预测Q值
                # 计算 Q 值
                q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                #----------------------Step 2 End ------------------------------------
                
                #----------------------Step 5 Begin ------------------------------------
                #Step 5: 计算 TD 目标值
                # 计算目标 Q 值
                next_q_values = target_dqn(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * ~dones
                #----------------------Step 5 End ------------------------------------
                
                # 计算损失
                loss = torch.nn.functional.mse_loss(q_values, target_q_values)

                # 更新网络
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 更新 epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # 每隔一定步数更新目标网络
        if episode % target_update_frequency == 0:
            target_dqn.load_state_dict(dqn.state_dict())

        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

# 训练模型
train_dqn(500)
# 保存模型
torch.save(dqn.state_dict(), 'dqn_lunarlander.pth')
