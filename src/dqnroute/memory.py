import random


class Memory:
    samples = []
    idx = 0

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append((self.idx, sample))
        self.idx += 1

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
