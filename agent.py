from collections import deque
import os
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
EPSILON_DECAY       = 0.9999
EPSILON_MIN         = 0.01
BATCH_SIZE          = 64
SYNC_NETWORK_RATE   = 1000


MEMORY_LENGTH       = 100000
SAMPLE_SIZE         = 1000

SHOULD_TRAIN        = False


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
        self.online_model = Network((1, 24, 32), 11, 3)
        self.target_model = Network((1, 24, 32), 11, 3, freeze=True)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.sync_network_rate = sync_network_rate
        self.batch_size = batch_size
        
        storage = LazyMemmapStorage(MEMORY_LENGTH)
        self.memory = TensorDictReplayBuffer(storage=storage)

        self.optimizer = optim.Adam(self.online_model.parameters(), lr)
        self.criterion = nn.MSELoss()

    def board(self, game):
        sizex = game.w // game.BLOCK_SIZE
        sizey = game.h // game.BLOCK_SIZE
        board = np.zeros((sizey, sizex), dtype=np.float32)
        for i in range(len(game.snake)):
            posx = int(game.snake[i].x) // game.BLOCK_SIZE
            posy = int(game.snake[i].y) // game.BLOCK_SIZE
            if posy < sizey and posx < sizex:
                board[posy][posx] = 1 if i == 0 else 0.7
        board[game.food.y // game.BLOCK_SIZE][game.food.x // game.BLOCK_SIZE] = -1
        return board

    
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
    
    def action(self, state, board):
        if np.random.random() < self.epsilon:
            index = np.random.randint(3)
        else:
            state_tensor = torch.tensor(np.array(state), dtype=torch.float) \
                .unsqueeze(0) \
                .to(self.online_model.device)
            board_tensor = torch.tensor(np.array(board), dtype=torch.float32) \
                .unsqueeze(0).unsqueeze(0) \
                .to(self.online_model.device)
            index = self.online_model(board_tensor, state_tensor).argmax().item()
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

        keys = ("state", "board", "action", "reward", "next_state", "next_board", "done")

        states, boards, actions, rewards, next_states, next_boards, dones = [samples[key] for key in keys]

        predicted_q_values = self.online_model(boards, states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        target_q_values = self.target_model(next_boards, next_states).max(dim=1)[0]
        target_q_values = rewards + self.gamma * target_q_values * (1 - dones.float())

        loss = self.criterion(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()

    def save_step(self, state, board, action, reward, new_state, new_board, done):
        self.memory.add(TensorDict({ 
                                    "state": torch.tensor(state, dtype=torch.float),
                                    "board": torch.tensor(board, dtype=torch.float32).unsqueeze(0),
                                    "action": torch.tensor(np.array(action), dtype=torch.long),
                                    "reward": torch.tensor(reward, dtype=torch.float),
                                    "next_state": torch.tensor(new_state, dtype=torch.float),
                                    "next_board": torch.tensor(new_board, dtype=torch.float32).unsqueeze(0),
                                    "done": torch.tensor(done)
                                }, batch_size=[]))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def load_model(self, path):
        self.online_model.load_state_dict(torch.load(path))
        self.target_model.load_state_dict(torch.load(path))

    
def play():
    max_score = 0
    total_score = 0
    agent = Agent()
    game = SnakeGame()

    if not SHOULD_TRAIN:
        file_name = 'model.pth'
        file_name = os.path.join('./model', file_name)
        agent.load_model(file_name)
        agent.epsilon = 0.0
        agent.epsilon_min = 0.0
        agent.epsilon_decay = 0.0

    while True:
        total_reward = 0
        done = False
        while not done:
            state = agent.state(game)
            board = agent.board(game)
            action = agent.action(state, board)
            reward, done, score = game.play_step(action)
            new_state = agent.state(game)
            new_board = agent.board(game)

            total_reward += reward
            if SHOULD_TRAIN:
                agent.save_step(state, board, action, reward, new_state, new_board, done)
                agent.train()

        game.reset()
        agent.game_counter += 1

        if score > max_score:
            max_score = score
            agent.online_model.save()

        total_score += score
        mean_score = total_score / agent.game_counter

        print('Game: ', agent.game_counter, 'Score: ', score, 'Max Score: ', max_score, 'Total reward: ', total_reward, 'Epsilon: ', agent.epsilon, 'Mean Score: ', mean_score)

if __name__ == '__main__':
    play()