import gym
import torch
import numpy as np
import torch.multiprocessing as mp

from torch import nn
from torch import optim
from torch.nn import functional as F


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim = 0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim = 0)
        c = F.relu(self.l3(y.detach()))
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic


def run_episode(worker_env, worker_model):
    pass


def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):
    pass


def worker(t, worker_model, counter, params):
    worker_env = gym.make('CartPole-v1')
    worker_env.reset()
    # parameters of the global model are shared
    worker_opt = optim.Adam(lr = 1e-4, params=worker_model.params())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        # pay attention to worker_model
        values, logprobs, rewards = run_episode(worker_env, worker_model)
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards)
        counter.value += 1




# main algorithm
MasterNode = ActorCritic()
MasterNode.share_memory() # method in nn.Module()
processes = []
params = {'epochs' : 1000, 'n_workers' : 2}
counter = mp.Value('i', 0)
for i in range(params['n_workers']):
    p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

for p in processes:
    p.terminate()
