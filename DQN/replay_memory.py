from collections import namedtuple
from random import sample

transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'terminal'])

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.contents = []
        self.idx = 0
    
    def add(self, s, a, r, s_new, terminal):
        if self.idx < self.capacity:
            self.contents.append(transition(s, a, r, s_new, terminal))
        else:
            self.contents[self.idx] = transition(s, a, r, s_new, terminal)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, num_samples):
        return sample(self.contents, num_samples)
    
    def length(self):
        return len(self.contents)
    
    def display(self):
        print(self.contents)