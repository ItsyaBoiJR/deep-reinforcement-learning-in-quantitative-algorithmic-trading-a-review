import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class TradingEnvironment:
    def __init__(self, price_series, initial_balance=1000):
        self.price_series = price_series
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array([self.balance, self.shares_held, self.price_series[self.current_step]])

    def step(self, action):
        if self.done:
            raise ValueError("Episode has ended. Please reset the environment.")

        current_price = self.price_series[self.current_step]
        reward = 0

        if action == 0:  # Buy
            self.shares_held += 1
            self.balance -= current_price
        elif action == 1:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
        elif action == 2:  # Hold
            pass

        self.current_step += 1
        if self.current_step >= len(self.price_series):
            self.done = True

        self.total_value = self.balance + self.shares_held * current_price
        reward = self.total_value - self.initial_balance

        return self._get_state(), reward, self.done

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            target_f = self.model(state)
            target_f = target_f.clone()
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate dummy price series
    price_series = np.sin(np.linspace(0, 100, 500)) * 10 + 100

    # Initialize environment and agent
    env = TradingEnvironment(price_series)
    state_size = 3  # [balance, shares_held, current_price]
    action_size = 3  # [buy, sell, hold]
    agent = DQNAgent(state_size, action_size)

    episodes = 100
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")
                break
        agent.replay(batch_size)