#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Remember, pyqt5 only works under Python3

import numpy as np
from rtlsdr import *
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
import threading
import time

# Parameters
ffts_to_avg = 300
fft_size = 512

# Initialize RTL-SDR
sdr = RtlSdr()

# configure RTL-SDR
sdr.sample_rate = 1.0e6
sdr.center_freq = 100e6
sdr.gain = 40

# house keeping
sample_period = 1.0/sdr.sample_rate
samples_per_loop = fft_size*ffts_to_avg*2
time_per_loop = samples_per_loop * sample_period

class Example(QWidget):
    def __init__(self):
        super().__init__()
        
        grid = QGridLayout()
        self.setLayout(grid)
        
        self.time_plot = pg.PlotWidget()
        self.time_plot_curve1 = self.time_plot.plot([]) 
        grid.addWidget(self.time_plot, 0, 0)

        self.setGeometry(300, 300, 300, 220)
        self.setWindowTitle('RTL-SDR Demo')
    
        self.show() # not blocking
       
        
def rx_thread(rtl, ex):
    while True:
        fft_running_avg = np.zeros(fft_size)
        rtl.read_samples(fft_size*ffts_to_avg)
        samples = rtl.read_samples(fft_size*ffts_to_avg) # causes 50% duty cycle in terms of processing
        t0 = time.time()
        for i in range(ffts_to_avg):
            fft = np.abs(np.fft.fft(samples[i*fft_size:(i+1)*fft_size]))
            fft_running_avg += fft
        results = 10.0*np.log10(np.fft.fftshift(fft_running_avg/ffts_to_avg))
        ex.time_plot_curve1.setData(results)
        print((time.time() - t0)/time_per_loop)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    rx_loop = threading.Thread(target=rx_thread, args=(sdr, ex))
    rx_loop.start()
    sys.exit(app.exec_())
    
    #rx_loop.join()
    sdr.close()
    
    
    
