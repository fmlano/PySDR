import numpy as np

# simple accumulator, returns True when it reaches the minimum samples specified and auto clears buffer
class accumulator:
    def __init__(self, min_samples):
        self.min_samples = min_samples
        self.samples = np.zeros(0, dtype=np.complex64)
        self.last_batch = False # signals to the next accumulate to clear the buffer
        
    def accumulate_samples(self, samples):
        if self.last_batch:
            self.samples = np.zeros(0, dtype=np.complex64)
            self.last_batch = False 
        self.samples = np.append(self.samples, samples)
        if len(self.samples) >= self.min_samples:
            self.last_batch = True
            return True
        else:
            return False
