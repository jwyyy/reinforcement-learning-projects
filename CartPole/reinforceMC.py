import gym
import torch
import numpy as np


# set up environment
# state, reward, done, info = env.step(action)
# the state is described by 4 variables
# reward is always 1 unless the pole falls down
env = gym.make('CartPole-v0')

# build a neural network to make actions
network = torch.nn.Sequential(
		torch.nn.Linear(4, 150),
		torch.nn.LeakReLU(),
		torch.nn.Linear(150, 2),
		torch.nn.Softmax())

learning_rate = 9e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def discount_rewards(rewards, gamma=0.99):
	r_len = len(rewards)
	disc_return = torch.pow(gamma, torch.arange(r_len).float()) * rewards
	disc_return /= disc_return.max() # normalize rewards to improve numerical stability, no necessary
	return disc_return

def loss_fn(probs, Gt):
	return -1 * torch.sum(r * torch.log(probs))


MAX_DURATION = 500 # the game might not end, so set the maximum length
MAX_EPISODES = 500
gamma = 0.99 # discount factor
score = []
for episode in range(MAX_EPISODES):
	curr_state = env.reset()
	done = False
	transactions = []

	# generate one episode
	for t in range(MAX_DURATION):
		action_prob = network(torch.from_numpy(curr_state).float())
		action = np.random.choice(np.array([0,1]), p = action_prob.data.numpy())
		prev_state = curr_state
		curr_state, _, done, info = env.step(action) # no need to record reward as long as the game continues, reward = 1
		transcations.append((prev_state, action, t+1))
		if done: break
	
	ep_len = len(transactions)
	score.append(ep_len) # reward = 1 at each step
	# the reward vector is flipped from the end to the begining
	reward_batch = torch.Tensor([r for (s,a,r) in transactions]).flip(dims(0,))
	disc_rewards = discount_rewards(reward_batch)
	state_batch = torch.Tensor([s for (s,a,r) in transactions])
	action_batch = torch.Tensor([a for (s,a,r) in transactions])
	pred_batch = model(state_batch) # compute the probability dist in each experienced state
	prob_batch = pred.batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()
	loss = loss_fn(prob_batch, disc_rewards)
	optimizer.zero_grad() # zero out previous computed gradients
	loss.backward() # compute gradients using backpropagation
	optimizer.step() # update parameters in the network

	
