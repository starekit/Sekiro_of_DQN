from torch import nn
import torch.nn.functional as f
from torch import optim
from collections import deque
from typing import List, Tuple
import random
import time
import torch
from pynput.keyboard import Controller


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, (5, 5))
        self.c2 = nn.Conv2d(6, 1, 5)
        self.fc1 = nn.Linear(225, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, inputs):
        x = f.max_pool2d(f.relu(self.c1(inputs)), 5)
        x = f.max_pool2d(f.relu(self.c2(x)), 5)
        x = x.view(-1, self.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class DQN(Net):
    def __init__(self, gamma, replay_buffer):
        super().__init__()
        self.gamma = gamma
        # self.action = action
        self.replay_buffer = replay_buffer
        self.loss = 0
        '''初始化model'''
        self.q = Net().cuda()
        self.target_q = Net().cuda()

        self.optimizer = optim.Adam(self.q.parameters(), lr=0.001)
        '''定义优化器'''

    def _compute_q_values(self, states: torch.Tensor) -> torch.Tensor:
        return self.q.forward(states)

    def _compute_target_q_values(self, next_states: torch.Tensor) -> torch.Tensor:
        return self.target_q.forward(next_states)

    @staticmethod
    def select_actions(transitions: List[Tuple]) -> torch.Tensor:
        return torch.tensor([transition[1] for transition in transitions], dtype=torch.int64).unsqueeze(-1).cuda()

    def update(self, transitions):
        """
        更新网络参数，依据给定的转换序列。

        参数:
        - transitions: 转换序列的列表，每个转换包含(start_state, action, next_state, reward)。

        概述:
        此方法首先解包转换序列并将状态数据移至CUDA设备，
        随后计算当前状态的Q值，选择动作，依据这些值更新网络参数。

        Updates the network's parameters based on a provided sequence of transitions.

        Parameters:
        - transitions: A list of transitions, each containing (start_state, action, next_state, reward).

        Summary:
        This method initially unpacks the transition sequence, moving state data to the CUDA device,
        then computes Q-values for current states, selects actions, and updates network parameters accordingly.
        """

        # 解析转换序列并迁移状态数据至CUDA设备
        # Unpack the transition sequence and migrate state data to the CUDA device
        states, actions, next_states, rewards = zip(*transitions)
        states = torch.stack(states).cuda()
        next_states = torch.stack(next_states).cuda()
        rewards = torch.tensor(rewards, dtype=torch.float32).cuda()

        # 计算当前状态下Q值
        # Compute Q-values for the current states
        q = self._compute_q_values(states)

        # 根据策略选择动作
        # Select actions based on the policy
        actions = self.select_actions(transitions)

        # 提取选中动作对应的Q值
        # Extract Q-values corresponding to the selected actions
        selected_q_values = torch.gather(q, 1, actions).squeeze(-1)

        # 计算下一状态的最大Q值作为目标Q值的一部分
        # Compute the maximum Q-value for the next states for the target Q-values
        target_q = self._compute_target_q_values(next_states)
        max_target_q = torch.max(target_q, dim=1)[0]

        # 结合奖励与折扣因子γ计算目标Q值
        # Compute target Q-values incorporating rewards and discount factor γ
        targets = rewards + self.gamma * max_target_q

        # 计算损失并执行反向传播更新网络参数
        # Compute loss and execute backpropagation to update network parameters
        loss = nn.MSELoss()(selected_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss.item()

    def q_target_update(self):
        self.target_q.load_state_dict(self.q.state_dict())


class ReplayBuffer:
    """
      初始化一个回放缓冲区对象。

      参数:
      buffer_max (int): 缓冲区的最大容量。

      属性:
      replay_buffer (deque): 用于存储经验的双端队列，超出最大容量时自动移除早期经验。
      buffer_max (int): 缓冲区设置的最大容量。
      """

    def __init__(self, buffer_max):
        self.replay_buffer = deque(maxlen=buffer_max)
        self.buffer_max = buffer_max

    def push(self, transition):
        self.replay_buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.replay_buffer, batch_size)

    def print(self):
        print(self.replay_buffer)

    def popleft(self):
        return self.replay_buffer.popleft()

    def __len__(self):
        return len(self.replay_buffer)


class AGENT(DQN):
    def __init__(self, gamma, replay_buffer):
        super().__init__(gamma, replay_buffer)

    def action_select(self, start_state):
        epsilon = 0.1

        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = torch.argmax(self.q.forward(start_state))

        return torch.tensor([action])  # 确保与torch.gather兼容


