from collections import namedtuple
from random import sample

transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'terminal'])

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.contents = []
        for i in range(capacity):
            self.contents.append(None)
        self.idx = 0
        self.length = 0
    
    def add(self, *args):
        self.contents[self.idx] = transition(*args)
        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0
        if self.length <= self.capacity:
            self.length += 1

    def sample(self, num_samples):
        return sample(self.contents, num_samples)
    
    def display(self):
        print(self.contents)