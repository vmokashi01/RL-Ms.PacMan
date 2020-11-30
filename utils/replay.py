from collections import deque
import random


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen = capacity)

    # add a transition 
    # state, action, reward, future state, done
    def add(self, s, a ,r, s2, d):
        self.memory.append((s, a, r, s2, d))

    # minibatch of a specified size
    def sample(self, batch_size, torch):
        minibatch = random.sample(self.memory, batch_size)
        S, A, R, S2, D = [], [], [], [], []

        for m in minibatch:
            s, a, r, s2, d = m
            S += [s]
            A += [a]
            R += [r]
            S2 += [s2]
            D += [d]

        return torch.to_f(S), torch.to_l(A), torch.to_f(R), torch.to_f(S2), torch.to_i(D)

    def __len__(self):
        return len(self.memory)