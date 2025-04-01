import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch
import numpy as np

class Network(nn.Module):
    def __init__(self, board_shape, extra_size, output_size, freeze=False, device=None):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(board_shape[0], 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(board_shape)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size + extra_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        if freeze:
            self._freeze()

        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, board, extra):
        conv_out = self.conv_layers(board)
        conv_out = conv_out.view(conv_out.size(0), -1)
        combined = torch.cat([conv_out, extra], dim=1)
        return self.fc_layers(combined)
    
    def save(self, file_name='model.pth'):
        if not os.path.exists('./model'):
            os.makedirs('./model')
        file_name = os.path.join('./model', file_name)
        torch.save(self.state_dict(), file_name)

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _freeze(self):
        for p in self.conv_layers.parameters():
            p.requires_grad = False
        for p in self.fc_layers.parameters():
            p.requires_grad = False
