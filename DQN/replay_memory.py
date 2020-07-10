from collections import namedtuple
from random import sample

transition = namedtuple('transition', ['state', 'q_values', 'action', 'reward', 'next_state', 'terminal'])

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.contents = []
        self.idx = 0
    
    def add(self, *args):
        if self.idx < self.capacity:
            self.contents.append(transition(*args))
        else:
            self.contents[self.idx] = transition(*args)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, num_samples):
        return sample(self.contents, num_samples)
    
    def length(self):
        return len(self.contents)
    
    def display(self):
        print(self.contents)