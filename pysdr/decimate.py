from __future__ import print_function # allows python3 print() to work in python2

import numpy as np
import time


# simple decimator
class decimate:
    def __init__(self, dec):
        self.dec = dec
        self.state = 0 # keeps track of how many elements need to be dropped at the beginning of the next batch
    def decimate(self, x):
        out = x[self.state::self.dec]
        self.state = -(len(x) - self.state) % self.dec
        return out
        

##############
# UNIT TESTS # 
##############
if __name__ == '__main__': # (call this script directly to run tests)

    #-----Test decimator-----
    x = np.random.random(5000) # signal
    decimation_factor = np.random.randint(1, 30) # random decimation factor between 1 and 29
    
    # simple method
    y = x[::decimation_factor]
    
    # batch type method using our decimator
    y2 = np.zeros(0)
    decimator1 = decimate(decimation_factor) # initialize decimator
    batch_size = np.random.randint(1000, 2000) # represents how many samples come in at the same time
    for i in range(len(x)/batch_size):
        x_input = x[i*batch_size:(i+1)*batch_size] # this line represents the incoming stream
        y2 = np.concatenate((y2, decimator1.decimate(x_input)))
    print("decimator test passed?", np.array_equal(y[0:len(y2)], y2)) # check if entire array is equal. dont include the very end because partial batches are not processed
   
