import gym
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import sys

sys.path.insert(1, '/Users/varunmokashi/Desktop/RL-PacMan/utils')
import utils.replay, utils.episode, utils.helper

# torch helper
t = utils.helper.TorchHelper()

# constants
DEVICE = t.device
OBS_N = 128                 # State space size
ACT_N = 9                   # Action space size
EPSILON = 1.0               # Starting epsilon
STEPS_MAX = 10000           # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1           # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64         # How many examples to sample per train step
GAMMA = 0.99                # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4        # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25           # Train for these many epochs every time
BUFSIZE = 10000             # Replay buffer size
EPISODES = 1000              # Total number of episodes to learn over
TEST_EPISODES = 10          # Test episodes

# create environment
env = gym.make('MsPacman-ram-v0')

# replay memory
memory = utils.replay.ReplayMemory(BUFSIZE)

# create CNN Q network - Q (state, action)
q = torch.nn.Sequential(
    torch.nn.Linear(OBS_N, 100), torch.nn.ReLU(),
    torch.nn.Linear(100, 50), torch.nn.ReLU(),
    torch.nn.Linear(50, ACT_N)
).to(DEVICE)

# epsilon-greedy policy
# we will uniformly randomly choose an action with probability epsilon 
# and use the greedy action with probability of 1 - epsilon 

def policy(env, obs):
    global EPSILON, STEPS, STEPS_MAX

    obs = t.to_f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
    # With probability EPSILON, choose a random action
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)

    # Rest of the time, choose argmax_a Q(s, a) 
    else:
        qvalues = q(obs)
        action = torch.argmax(qvalues).item()
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action

# training function
OPT = torch.optim.Adam(q.parameters(), lr = LEARNING_RATE) # Adam optimizer
LOSSFN = torch.nn.MSELoss() # mean square error loss function

def train(q, memory):

    global OPT, LOSSFN
    
    # sample a minbatch
    # each variable is a vector values
    S, A, R, S2, D = memory.sample(MINIBATCH_SIZE, t)

    # Get Q(s,a) for every state-action pair in the minibatch
    qvalues = q(S).gather(1, A.view(-1, 1)).squeeze()

    # Get max_a' Q(s', a') for every s' in the minibatch
    q2values = torch.max(q(S2), dim = 1).values

    targets = R + GAMMA * q2values * (1 - D)

    # detach the target, y
    loss = LOSSFN(targets.detach(), qvalues)

    # backpropogation
    OPT.zero_grad()
    loss.backward()
    OPT.step()

    return loss.item()


# play episodes
Rs = []
last25Rs = []
print("Training:")

# progress bar
pbar = tqdm.trange(EPISODES)
for ep in pbar:

    # play an episode, log the episodic reward
    S, A, R = utils.episode.play_ep_mem(env, policy, memory)
    Rs += [sum(R)]

    # train after collecting experience
    if ep >= TRAIN_AFTER_EPISODES:
        # train for specific number of training epochs
        # plot the loss
        for i in range(TRAIN_EPOCHS):
            train(q, memory)

    # show mean episodic reward for the last 25 eps
    last25Rs += [sum(Rs[-25:])/len(Rs[-25:])]
    pbar.set_description("Mean Episodic Reward for Previous 25: (%g)" % (last25Rs[-1]))

pbar.close()
print("Training finished!")

# plot
N = len(last25Rs)
plt.plot(range(N), last25Rs, 'b')
plt.savefig("images/pacmac-dqn.png")
print("Episodic reward plot saved")

# play test episodes
print("Testing:")
for ep in range(TEST_EPISODES):
    S, A, R = utils.episode.play_ep(env, policy, render = True)
    print("Episode %02d: R = %g" % (ep + 1, sum(R)))

env.close()