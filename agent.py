from collections import deque
import random
from snake_game import Point, Direction, SnakeGame
from model import Network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

#Hyperparameters
LR                  = 0.001
GAMMA               = 0.9
EPSILON             = 1
EPSILON_DECAY       = 0.95
EPSILON_MIN         = 0.01
BATCH_SIZE          = 64
SYNC_NETWORK_RATE   = 1000


MEMORY_LENGTH       = 100000
SAMPLE_SIZE         = 1000


class Agent:
    def __init__(self,
                 lr=LR,
                 gamma=GAMMA,
                 epsilon=EPSILON,
                 epsilon_decay=EPSILON_DECAY,
                 epsilon_min=EPSILON_MIN,
                 sync_network_rate=SYNC_NETWORK_RATE,
                 batch_size=BATCH_SIZE):
        self.game_counter = 0
        self.learn_step_counter = 0
        self.gamma = gamma # Discount rate
        self.lr = lr
        self.online_model = Network(11, 256, 3)
        self.target_model = Network(11, 256, 3, freeze=True)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.sync_network_rate = sync_network_rate
        self.batch_size = batch_size
        
        storage = LazyMemmapStorage(MEMORY_LENGTH)
        self.memory = TensorDictReplayBuffer(storage=storage)

        self.optimizer = optim.Adam(self.online_model.parameters(), lr)
        self.criterion = nn.MSELoss()

    
    def state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - game.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - game.BLOCK_SIZE)
        point_r = Point(head.x + game.BLOCK_SIZE, head.y)
        point_d = Point(head.x, head.y + game.BLOCK_SIZE)
        
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP
        dir_r = game.direction == Direction.RIGHT
        dir_d = game.direction == Direction.DOWN

        die_l = (dir_u and game.is_collision(point_l)) or \
                (dir_l and game.is_collision(point_d)) or \
                (dir_d and game.is_collision(point_r)) or \
                (dir_r and game.is_collision(point_u))
        die_s = (dir_u and game.is_collision(point_u)) or \
                (dir_l and game.is_collision(point_l)) or \
                (dir_d and game.is_collision(point_d)) or \
                (dir_r and game.is_collision(point_r)) 
        die_r = (dir_u and game.is_collision(point_r)) or \
                (dir_l and game.is_collision(point_u)) or \
                (dir_d and game.is_collision(point_l)) or \
                (dir_r and game.is_collision(point_d))
        
        apple_l = game.food.x < head.x
        apple_u = game.food.y < head.y
        apple_r = game.food.x > head.x
        apple_d = game.food.y > head.y

        state = [
            dir_l,
            dir_u,
            dir_r,
            dir_d,
            die_l,
            die_s,
            die_r,
            apple_l,
            apple_u,
            apple_r,
            apple_d
        ]

        return np.array(state, dtype=int)
    
    def action(self, state):
        if np.random.random() < self.epsilon:
            index = np.random.randint(3)
        else:
            state_tensor = torch.tensor(np.array(state), dtype=torch.float) \
                .unsqueeze(0) \
                .to(self.online_model.device)
            index = self.online_model(state_tensor).argmax().item()
        return index
    
    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_model.load_state_dict(self.online_model.state_dict())
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.sync_networks()

        self.optimizer.zero_grad()

        samples = self.memory.sample(self.batch_size).to(self.online_model.device)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_model(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        target_q_values = self.target_model(next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.criterion(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()

    def save_step(self, state, action, reward, new_state, done):
        self.memory.add(TensorDict({ 
                                    "state": torch.tensor(state, dtype=torch.float),
                                    "action": torch.tensor(np.array(action), dtype=torch.long),
                                    "reward": torch.tensor(reward, dtype=torch.float),
                                    "next_state": torch.tensor(new_state, dtype=torch.float),
                                    "done": torch.tensor(done)
                                }, batch_size=[]))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    
def play():
    max_score = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        total_reward = 0
        done = False
        while not done:
            state = agent.state(game)
            action = agent.action(state)
            reward, done, score = game.play_step(action)
            new_state = agent.state(game)

            total_reward += reward

            agent.save_step(state, action, reward, new_state, done)
            agent.train()

        game.reset()
        agent.game_counter += 1

        if score > max_score:
            max_score = score
            agent.online_model.save()

        print('Game: ', agent.game_counter, 'Score: ', score, 'Max Score: ', max_score, 'Total reward: ', total_reward)

if __name__ == '__main__':
    play()