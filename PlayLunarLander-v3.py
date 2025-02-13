import torch
import torch.nn as nn
import gymnasium as gym

# 初始化环境
env = gym.make("LunarLander-v3", render_mode="human")

# 定义 DQN 网络结构
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

# 初始化 DQN 网络
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
dqn = DQN(state_size, action_size)

# 加载训练好的模型参数
dqn.load_state_dict(torch.load('dqn_lunarlander.pth'))

# 设置模型为评估模式
dqn.eval()

# 与环境进行交互并玩游戏
state, info = env.reset()  # 获取初始状态
done = False
total_reward = 0

while not done:
    # 使用训练好的模型选择动作
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action = torch.argmax(dqn(state_tensor)).item()  # 选择 Q 值最大的动作

    # 执行动作并获得新状态、奖励
    next_state, reward, done, truncated, info = env.step(action)
    total_reward += reward

    # 更新状态
    state = next_state

    # 渲染游戏画面
    env.render()

print(f"Total Reward: {total_reward}")
env.close()
