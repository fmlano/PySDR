import numpy as np

# simple accumulator, returns True when it reaches the minimum samples specified and auto clears buffer
# assumes the min_samples is larger than the largest batch of samples recved by usrp
class accumulator:
    def __init__(self, min_samples): 
        self.min_samples = min_samples
        self.samples = np.zeros(min_samples, dtype=np.complex64) # stores the samples
        self.i = 0 # keeps track of where we are in the buffer
        self.last_batch = False # signals to the next accumulate to clear the buffer
        
    def accumulate_samples(self, samples):
        num_samples = samples.size
        if self.last_batch:
            self.last_batch = False
            self.samples[:] = np.zeros(self.min_samples, dtype=np.complex64) # clear buffer (mainly for debug at this point, no real reason to do it)
            self.samples[0:self.remainder.size] = self.remainder
            self.i = self.remainder.size
            self.samples[self.i:self.i+num_samples] = samples
            self.i += num_samples
            return False
        if self.i + num_samples <= self.min_samples: # if there is room for all the samples
            self.samples[self.i:self.i+num_samples] = samples
            self.i += num_samples
            return False
        else:  # if this batch will put us over
            self.samples[self.i:] = samples[0:self.min_samples - self.i]
            self.remainder = samples[self.min_samples - self.i:]
            self.last_batch = True
            return True
            



##############
# UNIT TESTS # 
##############
if __name__ == '__main__': # (call this script directly to run tests)
    accumulator1 = accumulator(25)
    x = np.arange(10)
    accumulator1.accumulate_samples(x)
    accumulator1.accumulate_samples(x)
    accumulator1.accumulate_samples(x)
    print accumulator1.samples
    accumulator1.accumulate_samples(x)
    print accumulator1.samples
