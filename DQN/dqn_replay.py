from collections import deque
from GridWorld import Gridworld
import numpy as np
import random
import torch


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
epsilon = 0.3

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

epochs = 5000
losses = []
mem_size = 1000
batch_size = 200
replay = deque(maxlen = mem_size)
max_moves = 50
h = 0

for i in range(epochs):
    game = Gridworld(size = 4, mode = 'random')
    state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    state1 = torch.from_numpy(state1_).float()
    status = 1
    mov = 0
    while status == 1:
        mov += 1
        qval = model(state1)
        qval_ = qval.data.numpy()
        if random.random() < epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)

        action = action_set[action_]
        game.makeMove(action)
        state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
        state2 = torch.from_numpy(state2_).float()
        reward = game.reward()
        done = True if reward > 0 else False

        exp = (state1, action_, reward, state2, done)
        replay.append(exp)
        state1 = state2

        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = torch.cat([s1 for (s1, _, _, _, _) in minibatch])
            action_batch = torch.Tensor([a for (_, a, _, _, _) in minibatch])
            reward_batch = torch.Tensor([r for (_, _, r, _, _) in minibatch])
            state2_batch = torch.cat([s2 for (_, _, _, s2, _) in minibatch])
            done_batch = torch.Tensor([d for (_, _, _, _, d) in minibatch])

            Q1 = model(state1_batch)
            with torch.no_grad():
                Q2 = model(state2_batch)
            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
            # what does gather() function do?
            # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
            # X is the batch output = Q(s,a) for all actions a
            # gather() collects the Q(s,a) corresponding to a specific action
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        if reward != -1 or mov > max_moves:
            status = 0
            mov = 0

        # not decreasing epsilon

    print("epoch =", i, ", loss = ", losses[-1] if len(losses) else None)
losses = np.array(losses)


# test the trained model
max_games = 1000
wins = 0

for i in range(max_games):
    win = test_model(model, mode = 'random', display=False)
    if win: wins += 1

win_prec = float(wins) / float(max_games)
print("Games played: {0}, # of wins: {1}".format(max_games, wins))
print("Win percentage: {}".format(win_prec))
