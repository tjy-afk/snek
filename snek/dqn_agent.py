import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Hyperparameters
HIDDEN_SIZE = 128 # Controls the number of neurons in the hidden layers of the neural network.
GAMMA = 0.99 # Controls how much the agent values future rewards.
EPSILON_START = 1.0 # Determines the initial exploration rate.
EPSILON_DECAY = 0.995 # Determines the rate at which the exploration rate decays over time.
EPSILON_MIN = 0.01 # Sets the minimum exploration rate.
LEARNING_RATE = 0.001  # Determines the learning rate for the optimizer.
BATCH_SIZE = 128  # Determines how stable the learning process is during training.
MEMORY_SIZE = 10000 # Affects how much of the agent’s history influences its current learning.

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.batch_size = BATCH_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Network
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            action = torch.LongTensor([action]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            # Compute target
            with torch.no_grad():
                target = reward + (1.0 - done) * self.gamma * torch.max(self.q_network(next_state))
            
            # Compute Q-value for the selected action
            current_q = self.q_network(state)[action]
            
            # Compute loss
            loss = nn.functional.mse_loss(current_q, target)
            
            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.q_network.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.q_network.state_dict(), name)
