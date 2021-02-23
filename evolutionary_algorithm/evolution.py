# reference: Deep RL in action Chapter 6
#            Alternative optimization methods: evolutionary algorithm

import gym
import numpy as np
import torch 


def model(x, unpacked_params):
    l1, b1, l2, b2, l3, b3 = unpacked_params
    y = torch.nn.functional.linear(x, l1, b1)
    y = torch.relu(y)
    y = torch.nn.functional.linear(y, l2, b2)
    y = torch.relu(y)
    y = torch.nn.functional.linear(y, l3, b3)
    y = torch.log_softmax(y, dim = 0)
    return y


# parameters are input as a flattened vector
# this makes mutation and re-combination in the evolution algorithm easier to implement
def unpack_params(params, layers = [(25, 4), (10, 25), (2, 10)]):
    unpacked_params = []
    e = 0
    for i, l in enumerate(layers):
        s, e = e, e + np.prod(l)
        # like re-shape the vector of weights into a matrix
        weights = params[s:e].view(l)
        s, e = e, e + l[0]
        bias = params[s:e]
        unpacked_params.extend([weights, bias])
    return unpacked_params


def spawn_population(N = 50, size = 407):
    pop = []
    for i in range(N):
        vec = torch.randn(size) / 2.0
        fit = 0
        p = {'params' : vec, 'fitness' : fit}
        pop.append(p)
    return pop

