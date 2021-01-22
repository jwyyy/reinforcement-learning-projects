from GridWorld import Gridworld
import torch
import numpy as np
import random

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
epsilon = 1.0

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# learning goal: Q(s_t, a) = E[ r_t+1 + gamma max_a Q(s_t+1, a) | s_t, a_t]
# TD-1, bootstrap 1-step ahead

epochs = 1000
losses = [] # store training process
for i in range(epochs):
    game = Gridworld(size = 4, mode = 'static')
    # an extra noise is added to the board
    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state1 = torch.from_numpy(state_).float()
    # status = 0 means end of the current game
    status = 1
    while status == 1:
        qval = model(state1)
        qval_ = qval.data.numpy()
        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)

        action = action_set[action_]
        game.makeMove(action)

        # t+1: 1 step forward
        # compute r_t+1 + gamma * max_a Q(s_t+1, a)
        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state2 = torch.from_numpy(state2_).float()
        # this reward is r_t+1
        reward = game.reward()

        # the following post explains the function of no_grad()
        # torch.no_grad() vs model.eval()
        # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/2
        with torch.no_grad():
            newQ = model(state2.reshape(1,64))
        maxQ = torch.max(newQ)

        if reward == -1:
            Y = reward + gamma * maxQ
        else:
            Y = reward

        # the gradient won't pass through reward + gamma Q_max
        Y = torch.Tensor([Y]).detach()
        X = qval.squeeze()[action_]
        loss = loss_fn(X, Y)

        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        state1 = state2
        if reward != -1:
            status = 0
    print("epoch =", i, ", loss = ", losses[-1])
    if epsilon > 0.1:
        epsilon -= (1/epochs)


def test_model(model, mode = 'static', display = True):
    i = 0
    test_game = Gridworld(mode = mode)
    state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
    state = torch.from_numpy(state_).float()
    if display:
        print("Initial State:")
        print(test_game.display())

    status = 1
    while status == 1:
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        action = action_set[action_]
        if display:
            print("Move #: %s; Taking action: %s" % (i, action))
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 10.0
        state = torch.from_numpy(state_).float()
        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display: print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display: print("Game LOST! Reward: %s" % (reward,))
        i += 1
        if i > 15:
            if display: print("Game lost; too many moves.")
            break
    win = True if status == 2 else False
    return win


# test the learned model with static model
test_model(model, 'static')
