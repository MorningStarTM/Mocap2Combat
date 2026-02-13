import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim





class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.Tanh):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers += [nn.Linear(prev, output_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



