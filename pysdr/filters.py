# The point of these filters is not to "implement filters" per se, it's to use existing implementations of filters
#    but provide them in a "streaming" manner, where batches of samples go through them without the developer
#    having to worry about discontinuities and such.  

import numpy as np
import time
from scipy import signal

# a np.convolve based filter
class fir_filter:
    def __init__(self, taps):
        self.taps = taps
        self.previous_batch = np.zeros(len(self.taps) - 1, dtype=np.complex128) # holds end of previous batch, this is the "state" essentially

    def filter(self, x):
        out = np.convolve(np.concatenate((self.previous_batch, x)), self.taps, mode='valid')
        self.previous_batch[:] = x[-(len(self.taps) - 1):] # the last portion of the batch gets saved for the next iteration #FIXME if batches become smaller than taps this won't work
        return out

# an fft based filter (currently sux)
class fft_filter:
    def __init__(self, taps):
        self.taps = taps
        self.previous_batch = np.zeros(len(self.taps) - 1, dtype=np.complex128) # holds end of previous batch, this is the "state" essentially
    def filter(self, x):
        out = signal.fftconvolve(np.concatenate((self.previous_batch, x)), self.taps, mode='valid')
        self.previous_batch[:] = x[-(len(self.taps) - 1):] # the last portion of the batch gets saved for the next iteration #FIXME if batches become smaller than taps this won't work
        return out


##############
# UNIT TESTS # 
##############
if __name__ == '__main__': # (call this script directly to run tests)

    #-----Test for FIR filter-----
    x = np.random.randn(20000) + 1j*np.random.randn(20000) # signal
    taps = np.random.rand(100)

    # simple method of filtering
    y = np.convolve(x, taps, mode='valid') # valid means the convolution product is only given for points where the signals overlap completely
                                           # the output size is the len(x) - len(taps) + 1 (the taps are like a delay)                                      
    # now using our stream-based FIR filter
    y2 = np.zeros(0) # empty array that will contain outputs concatenated
    y3 = np.zeros(0)
    batch_size = 2000 # represents how many samples come in at the same time
    test_filter = fir_filter(taps) # initialize filter (normally we would need to do pysdr.fir_filter(taps) but this is in the same file)
    test_filter2 = fft_filter(taps)
    for i in range(len(x)/batch_size):
        x_input = x[i*batch_size:(i+1)*batch_size] # this line represents the incoming stream
        #start = time.time()
        filter_output = test_filter.filter(x_input) # run the filter
        #print 'It took', time.time()-start, 'seconds.'
        y2 = np.concatenate((y2, filter_output)) # add output to our log
        filter_output = test_filter2.filter(x_input) # run the filter
        y3 = np.concatenate((y3, filter_output))
        
    # check if they are equal
    y2 = y2[len(taps)-1:] # get rid of the beginning that was computed using zeros, in order to make it equal to simple method (this wont matter in real apps, its just a transient thing)
    print "fir_filter test passed? ", np.array_equal(y, y2) # check if entire array is equal
    y3 = y3[len(taps)-1:] 
    print "fft_filter test passed? ", np.allclose(y, y3, rtol=1e-3) # check if entire array is ROUGHLY equal

    print sum((y == y2).astype(int))/float(len(y))
    

