import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, freeze=False):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        if freeze:
            self._freeze()

        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        return self.network(x)
    
    def save(self, file_name='model.pth'):
        if not os.path.exists('./model'):
            os.makedirs('./model')
        file_name = os.path.join('./model', file_name)
        torch.save(self.state_dict(), file_name)

    def _freeze(self):
        for p in self.network.parameters():
            p.requires_grad = False
