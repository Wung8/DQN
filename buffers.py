import random

class Uniform_replay_buffer():

    # s, a, r, s', mask

    def __init__(self, buffer_size, sample_size):
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.replay_buffer = []

    # hist: [ state, action, max_next_q, r, next_state, next_valid_actions ]
    def add(self, hist):
        to_add = list(zip(*hist))
        self.replay_buffer = to_add + self.replay_buffer
        self.replay_buffer = self.replay_buffer[:self.buffer_size]

    def sample(self):
        if self.sample_size >= len(self.replay_buffer):
            samples = self.replay_buffer
        else:
            idxs = {*random.sample(range(len(self.replay_buffer)), self.sample_size)}
            samples = [item for i,item in enumerate(self.replay_buffer) if i in idxs]
        return samples
