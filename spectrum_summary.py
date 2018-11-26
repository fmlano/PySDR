import numpy as np
from scipy import signal
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import QSize, pyqtSlot
import pyqtgraph as pg





from datetime import datetime, timedelta
import sys
import time
import threading
import logging
import numpy as np
import uhd
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


# Parameters
start_freq = 300e6 # (center freq)
stop_freq = 500e6 #3e9 # (center freq)
rx_rate = 20e6
fft_size = 2048
ffts_to_avg = 200

rx_freq = 100e6

CLOCK_TIMEOUT = 1000  # 1000mS timeout for external clock locking
INIT_DELAY = 0.05  # 50mS initial delay before transmit


def benchmark_rx_rate(usrp, rx_streamer, timer_elapsed_event, rx_statistics, win):
    """Benchmark the receive chain"""
    logger.info("Testing receive rate {:.3f} Msps on {:d} channels".format(
        usrp.get_rx_rate()/1e6, rx_streamer.get_num_channels()))

    # Make a receive buffer
    num_channels = rx_streamer.get_num_channels()
    max_samps_per_packet = rx_streamer.get_max_num_samps()
    # TODO: The C++ code uses rx_cpu type here. Do we want to use that to set dtype?
    recv_buffer = np.empty((num_channels, max_samps_per_packet), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    # Craft and send the Stream Command
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = (num_channels == 1)
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
    results = np.empty(0, dtype=float)
    running_mean = np.zeros(fft_size, dtype=float)
    running_max  = -99.9 * np.ones(fft_size, dtype=float)
    while not timer_elapsed_event.is_set():
        try:
            num_rx_samps = rx_streamer.recv(recv_buffer, metadata) * num_channels
            FFT = np.abs(np.fft.fft(recv_buffer[0], fft_size))
            running_mean += FFT
            running_max = np.maximum(running_max, FFT)
            i += 1
            if i == ffts_to_avg:
                mean = 10.0*np.log10(np.fft.fftshift(running_mean/ffts_to_avg))
                maxx = 10.0*np.log10(np.fft.fftshift(running_max))
                win.plotDataItem1.setData(np.arange(len(mean)), mean) # mean
                #win.plotDataItem2.setData(np.arange(len(mean)), maxx) # max
                # reset everything
                running_mean = np.zeros(fft_size, dtype=float)
                running_max  = -99.9 * np.ones(fft_size, dtype=float)
                i = 0

            

            
            
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
            stream_cmd.stream_now = (num_channels == 1)
            rx_streamer.issue_stream_cmd(stream_cmd)
        elif metadata.error_code == uhd.types.RXMetadataErrorCode.timeout:
            logger.warning("Receiver error: %s, continuing...", metadata.strerror())
            num_rx_timeouts += 1
        else:
            logger.error("Receiver error: %s", metadata.strerror())
            logger.error("Unexpected error on receive, continuing...")


    # After we get the signal to stop, issue a stop command
    rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))




'''
class MyWidget(pg.GraphicsWindow):
    def __init__(self, parent=None):
        super(MyWidget, self).__init__(parent=parent)


'''

        
        
def main():
    app = QtWidgets.QApplication([]) # has to be called (just once) at the beginning of a qt app

    pg.setConfigOptions(antialias=False) # True seems to work as well

    win = pg.GraphicsWindow(title="window title")
    win.resize(1500,600)
    win.setLayout(QtWidgets.QVBoxLayout()) # create and set a layout that will tile plots horizontally
    
    # create first plot
    win.plotItem1 = win.addPlot(title="Plot 1 Title") # adds a plot
    win.plotItem1.setRange(yRange=[-20, -5])
    win.plotDataItem1 = win.plotItem1.plot([], pen=(0, 255, 0)) # adds a curve to the plot
    #win.plotDataItem2 = win.plotItem1.plot([], pen=(255, 0, 0)) # adds a curve to the plot

    
    win.show()
    win.raise_()
    
    # Setup a usrp device
    usrp = uhd.usrp.MultiUSRP("num_recv_frames=1000") # USRP args go here, see https://files.ettus.com/manual/page_transport.html and https://files.ettus.com/manual/page_configuration.html#config_devaddr

    # Always select the subdevice first, the channel mapping affects the other settings
    #usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(args.rx_subdev))


    rx_channels = [0]


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
    
    st_args = uhd.usrp.StreamArgs("fc32", "sc16") # host/computer format, otw format
    st_args.channels = rx_channels
    rx_streamer = usrp.get_rx_stream(st_args)
    print "max samps per buffer: ", rx_streamer.get_max_num_samps() # affected by recv_frame_size
    rx_thread = threading.Thread(target=benchmark_rx_rate, args=(usrp, rx_streamer, quit_event, rx_statistics, win))
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
    for i in range(100):
        rx_streamer.recv(recv_buffer, metadata)
    del usrp # stops USRP cleanly, otherwise we sometimes get libusb errors when starting back up
        
    
    #print_statistics(rx_statistics)

    return True


if __name__ == "__main__":
    # Setup the logger with our custom timestamp formatting
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    logger.addHandler(console)

    # Vamos, vamos, vamos!
    sys.exit(not main())
    pg.QtGui.QApplication.exec_() # you MUST put this at the end
    
    
