from GridWorld import Gridworld
import torch
import numpy as np

model = torch.nn.Sequential(
    torch.nn.Linear(64, 150),
    torch.nn.ReLU(),
    torch.nn.Linear(150, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 4)
)
# goal is to learn the optimal Q-function (defined in Bellman equation)
loss_fn = torch.nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

gamma = 0.9
alpha = 0.2
