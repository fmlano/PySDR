#!/usr/bin/env python
# Initially based on  uhd/host/examples/python/benchmark_rate.py
# This app receives a stream off the USRP and stores it in a circular buffer.  pieces of IQ can be requested via a message
# It also monitors for overflows, to make sure we aren't losing samples

# Benchmarking:
# once I added num_recv_frames=1000 I was able to get up to 56M without dropped samples or overruns!

from datetime import datetime, timedelta
import sys
import time
import threading
import logging
import numpy as np
import uhd
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton
from PyQt5.QtCore import QRect
import pyqtgraph as pg


# Parameters
rx_rate = 10e6
rx_freq = 100e6
rx_channels = [0]
duration = 2 # in seconds
fft_size = 512
num_rows = 100
num_to_avg = 50
chunk_decimation_factor = 10 # so that we dont processes 100% of samples

CLOCK_TIMEOUT = 1000  # 1000mS timeout for external clock locking
INIT_DELAY = 0.05  # 50mS initial delay before transmit





def benchmark_rx_rate(usrp, rx_streamer, timer_elapsed_event, rx_statistics, win):
    """Benchmark the receive chain"""
    logger.info("Testing receive rate {:.3f} Msps on {:d} channels".format(usrp.get_rx_rate()/1e6, 1))

    # Make a receive buffer
    max_samps_per_packet = rx_streamer.get_max_num_samps()
    # TODO: The C++ code uses rx_cpu type here. Do we want to use that to set dtype?
    recv_buffer = np.empty((1, max_samps_per_packet), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    # Craft and send the Stream Command
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    stream_cmd.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + INIT_DELAY)
    rx_streamer.issue_stream_cmd(stream_cmd)

    # To estimate the number of dropped samples in an overflow situation, we need the following
    # On the first overflow, set had_an_overflow and record the time
    # On the next ERROR_CODE_NONE, calculate how long its been since the recorded time, and use the
    #   tick rate to estimate the number of dropped samples. Also, reset the tracking variables
    had_an_overflow = False
    last_overflow = uhd.types.TimeSpec(0)
    # Setup the statistic counters
    num_rx_dropped = 0
    num_rx_overruns = 0
    num_rx_seqerr = 0
    num_rx_timeouts = 0
    num_rx_late = 0

    rate = usrp.get_rx_rate()
    # Receive until we get the signal to stop
    i = 0
    ii = 0
    rx_streamer.recv(recv_buffer, metadata) # to see around what level we are receiving at, to init waterfall 2d array
    avg_value = np.mean(10.0*np.log10(np.abs(np.fft.fft(recv_buffer[0], fft_size))))
    data = avg_value * np.ones((fft_size, num_rows))
    running_avg = np.zeros(fft_size)
    first_time = True
    f = np.linspace(rx_freq - rx_rate/2.0, rx_freq + rx_rate/2.0, fft_size) / 1e6
    t = np.arange(500)/rx_rate*1e6
    while not timer_elapsed_event.is_set():
        try:
            rx_streamer.recv(recv_buffer, metadata)
            i += 1
            if i == chunk_decimation_factor: # used to chunck decimate, so that we dont have to process 100% of samples...
                running_avg += np.abs(np.fft.fft(recv_buffer[0], fft_size))
                i = 0
                ii += 1
                if ii == num_to_avg:
                    win.time_plot_curve_i.setData(t, np.real(recv_buffer[0][0:500])) # time plot
                    win.time_plot_curve_q.setData(t, np.imag(recv_buffer[0][0:500])) # time plot
                    
                    fft = 10.0*np.log10(np.fft.fftshift(running_avg/num_to_avg))
                    
                    win.fft_plot_curve_fft.setData(f, fft) # FFT plot
                
                    # create waterfall
                    data[:] = np.roll(data, 1, axis=1) # shifts waterfall 1 row
                    data[:,0] = fft # fill last row with new fft results
                        
                    # Display waterfall
                    win.imageitem.setImage(data) # auto ranges by default
                    
                    running_avg = np.zeros(fft_size)
                    ii = 0
                    
                    if first_time:
                        first_time = False
                        # time and freq adjustments
                        win.time_plot.autoRange()
                        win.fft_plot.autoRange()
                        # waterfall adjustments
                        samples_per_row = len(recv_buffer[0]) * num_to_avg * chunk_decimation_factor / rx_rate
                        win.imageitem.translate((rx_freq - rx_rate/2.0)/1e6, 0)
                        win.imageitem.scale(rx_rate/fft_size/1e6, samples_per_row)
                        win.waterfall.autoRange()
        
                
        except RuntimeError as ex:
            logger.error("Runtime error in receive: %s", ex)
            return

        # Handle the error codes
        if metadata.error_code == uhd.types.RXMetadataErrorCode.none:
            # Reset the overflow flag
            if had_an_overflow:
                had_an_overflow = False
                num_rx_dropped += uhd.types.TimeSpec(
                    metadata.time_spec.get_real_secs() - last_overflow.get_real_secs()
                ).to_ticks(rate)
        elif metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
            had_an_overflow = True
            last_overflow = metadata.time_spec
            # If we had a sequence error, record it
            if metadata.out_of_sequence:
                num_rx_seqerr += 1
                logger.warning("Detected RX sequence error.")
            # Otherwise just count the overrun
            else:
                num_rx_overruns += 1
        elif metadata.error_code == uhd.types.RXMetadataErrorCode.late:
            logger.warning("Receiver error: %s, restarting streaming...", metadata.strerror())
            num_rx_late += 1
            # Radio core will be in the idle state. Issue stream command to restart streaming.
            stream_cmd.time_spec = uhd.types.TimeSpec(
                usrp.get_time_now().get_real_secs() + INIT_DELAY)
            stream_cmd.stream_now = True
            rx_streamer.issue_stream_cmd(stream_cmd)
        elif metadata.error_code == uhd.types.RXMetadataErrorCode.timeout:
            logger.warning("Receiver error: %s, continuing...", metadata.strerror())
            num_rx_timeouts += 1
        else:
            logger.error("Receiver error: %s", metadata.strerror())
            logger.error("Unexpected error on receive, continuing...")

    # After we get the signal to stop, issue a stop command
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))




class Example(QWidget):
    def __init__(self):
        super().__init__()
        
        # Set up layout
        grid = QGridLayout()
        self.setLayout(grid)
        
        pg.setConfigOptions(antialias=False) # True seems to work as well

        # Create pushbutton that resets the views
        self.button = QPushButton('Reset All Zooms', self)
        self.button.clicked.connect(self.handleButton)
        grid.addWidget(self.button, 0, 0)
        
        
        # create time plot
        self.time_plot = pg.PlotWidget(labels={'left': 'Amplitude', 'bottom': 'Time [microseconds]'}, enableMenu=False)
        self.time_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.time_plot.setMouseEnabled(x=False, y=True)
        self.time_plot_curve_i = self.time_plot.plot([]) 
        self.time_plot_curve_q = self.time_plot.plot([]) 
        grid.addWidget(self.time_plot, 1, 0)
        
        # create fft plot
        self.fft_plot = pg.PlotWidget(labels={'left': 'PSD', 'bottom': 'Frequency [MHz]'}, enableMenu=False)
        self.fft_plot.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.fft_plot.setMouseEnabled(x=False, y=True)
        self.fft_plot_curve_fft = self.fft_plot.plot([]) 
        grid.addWidget(self.fft_plot, 2, 0)
        
        # Create waterfall plot
        self.waterfall = pg.PlotWidget(labels={'left': 'Time [s]', 'bottom': 'Frequency [MHz]'}, enableMenu=False)
        self.waterfall.getPlotItem().getViewBox().translateBy(x=10.0)
        self.imageitem = pg.ImageItem()
        self.waterfall.addItem(self.imageitem)
        self.waterfall.setMouseEnabled(x=False, y=False)
        grid.addWidget(self.waterfall, 3, 0)
  
        self.setGeometry(300, 300, 300, 220) # window placement and size
        self.setWindowTitle('RTL-SDR Demo')
    
        self.show() # not blocking
        
    def handleButton(self):
        self.time_plot.autoRange()
        self.fft_plot.autoRange()
                    
            
if __name__ == "__main__":
    # Setup the logger with our custom timestamp formatting
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    logger.addHandler(console)

    app = QApplication(sys.argv)
    ex = Example()

    # Setup a usrp device
    usrp = uhd.usrp.MultiUSRP("num_recv_frames=1000") # USRP args go here, see https://files.ettus.com/manual/page_transport.html and https://files.ettus.com/manual/page_configuration.html#config_devaddr

    # Always select the subdevice first, the channel mapping affects the other settings
    #usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(args.rx_subdev))

    rx_channels = [0] # receive on the first channel

    usrp.set_time_now(uhd.types.TimeSpec(0.0))
    
    threads = []
    # Make a signal for the threads to stop running
    quit_event = threading.Event()
    # Create a dictionary for the RX statistics
    # Note: we're going to use this without locks, so don't access it from the main thread until
    #       the worker has joined
    rx_statistics = {}
    
    # Spawn the receive test thread
    usrp.set_rx_rate(rx_rate)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(rx_freq), 0)
    usrp.set_rx_gain(50, 0)
    st_args = uhd.usrp.StreamArgs("fc32", "sc16") # host/computer format, otw format
    st_args.channels = rx_channels
    rx_streamer = usrp.get_rx_stream(st_args)
    print("max samps per buffer: ", rx_streamer.get_max_num_samps()) # affected by recv_frame_size
    rx_thread = threading.Thread(target=benchmark_rx_rate, args=(usrp, rx_streamer, quit_event, rx_statistics, ex))
    threads.append(rx_thread)
    rx_thread.start()
    rx_thread.setName("bmark_rx_stream")

    app.exec_() # this is the blocking line from the pyqtgrpah example im using
    
    # Interrupt and join the threads
    quit_event.set()
    for thr in threads:
        thr.join()
         
    # These three lines make sure that when you close the Qt window, the USRP closes cleanly
    recv_buffer = np.empty((1, rx_streamer.get_max_num_samps()), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()
    for i in range(10):
        rx_streamer.recv(recv_buffer, metadata)
    del usrp # stops USRP cleanly, otherwise we sometimes get libusb errors when starting back up
        
    
    
