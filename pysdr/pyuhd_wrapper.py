from __future__ import print_function # allows python3 print() to work in python2

from uhd import libpyuhd
import numpy as np
import sys

# --- install pyuhd as follows --- 
# git clone https://github.com/EttusResearch/uhd.git
# git checkout python-api
# cd host
# mkdir build
# cmake -DCMAKE_INSTALL_PREFIX=~/pybombs-target ../
# make -j 4
# sudo make install

# Even though pyuhd is a wrapper for UHD, our wrapper of a wrapper makes it a bit easier to use and i havent seen a performance loss
class usrp_source(libpyuhd.usrp.multi_usrp):
    def __init__(self, usrp_args=''):
        super(usrp_source, self).__init__(usrp_args)
    
    def set_samp_rate(self, samp_rate):
        self.set_rx_rate(samp_rate, 0)
        
    def set_center_freq(self, center_freq):
        self.set_rx_freq(libpyuhd.types.tune_request(center_freq), 0) # apparently you have to do the tune request function
        
    def set_gain(self, gain):
        self.set_rx_gain(gain, 0)    
        
    def prepare_to_rx(self):
        st_args = libpyuhd.usrp.stream_args("fc32", "sc16")
        st_args.channels = [0] # channel
        self.metadata = libpyuhd.types.rx_metadata()
        try:
            self.streamer = self.get_rx_stream(st_args) # keep the streamer an object of usrp
        except RuntimeError:
            print("libusb got screwed up- try unplugging and replugging in the USRP")
            sys.exit("hit control-C to quit script")
        buffer_samps = self.streamer.get_max_num_samps()
        print("max_num_samps:", buffer_samps)
        self.recv_buffer = np.zeros(buffer_samps, dtype=np.complex64) # buffer is also an object of usrp
        stream_cmd = libpyuhd.types.stream_cmd(libpyuhd.types.stream_mode.start_cont)
        stream_cmd.stream_now = True
        self.streamer.issue_stream_cmd(stream_cmd)
        
    def recv(self):
        num_samps = self.streamer.recv(self.recv_buffer, self.metadata) # receive samples! returns number of samples
        #if num_samps == 0:
        #    print("APPARENTLY ITS NOT A BLOCKING FUNCTION!")
        # check if there were any errors        
        if self.metadata.error_code != libpyuhd.types.rx_metadata_error_code.none:
            print(self.metadata.strerror())
        # return the samples
        return self.recv_buffer
