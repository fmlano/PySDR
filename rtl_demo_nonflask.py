#!/usr/bin/env python

# to run this app use: bokeh serve rtl_demo_nonflask.py --show

import pysdr # our python package

import numpy as np
import time 

from scipy import signal # includes filters and filter designers

from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot, Spacer, widgetbox
from bokeh.models import Select, TextInput
from bokeh.util.browser import view # utility to Open a browser to view the specified location.

from rtlsdr import RtlSdr

from multiprocessing import Process, Manager 

# Parameters
fft_size = 512               # output size of fft, the input size is the samples_per_batch
waterfall_samples = 100      # number of rows of the waterfall
samples_per_batch = 256*1024 # num of samples that we process at a time
samples_in_time_plots = 500  # should be less than samples_per_batch

# RTL-SDR stuff
sdr = RtlSdr()
sdr.sample_rate = 2.048e6  # Hz
sdr.center_freq = 101.1e6  # Hz
sdr.freq_correction = 60   # PPM
sdr.gain = sdr.get_gains()[-1]/10.0 # highest gain to start with

# Set up the shared buffer between threads (using multiprocessing's Manager).  it is global
manager = Manager()
shared_buffer = manager.dict() # there is also an option to use a list
shared_buffer['waterfall'] = np.ones((waterfall_samples, fft_size))*-100.0 # waterfall buffer
shared_buffer['psd'] = np.zeros(fft_size) # PSD buffer
shared_buffer['i'] = np.zeros(samples_in_time_plots) # I buffer (time domain)
shared_buffer['q'] = np.zeros(samples_in_time_plots) # Q buffer (time domain)
shared_buffer['stop-signal'] = False # used to signal RTL to stop (when it goes true)
shared_buffer['utilization'] = 0.0 # float between 0 and 1, used to store how the process_samples is keeping up

# create a streaming-type FIR filter (this should act the same as a FIR filter block in GNU Radio)
taps = signal.firwin(numtaps=100, cutoff=500e3, nyq=sdr.sample_rate)
prefilter = pysdr.fir_filter(taps)

# Function that processes each batch of samples that comes in (currently, all DSP goes here)
def process_samples(samples, rtlsdr_obj):
    startTime = time.time()
    samples = prefilter.filter(samples) # 0.01s
    PSD = 10.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples, fft_size)/float(fft_size)))**2) # calcs PSD
    waterfall = shared_buffer['waterfall'] # pull waterfall from buffer
    waterfall[:] = np.roll(waterfall, -1, axis=0) # shifts waterfall 1 row
    waterfall[-1,:] = PSD # fill last row with new fft results
    shared_buffer['waterfall'] = waterfall # you have to copy it back into the manager_list
    shared_buffer['psd'] = PSD # overwrites whatever was in psd buffer, so that the GUI uses the most recent one when it goes to refresh itself
    shared_buffer['i'] = np.real(samples[0:samples_in_time_plots]) # i buffer
    shared_buffer['q'] = np.imag(samples[0:samples_in_time_plots]) # q buffer
    # if the change-gain or change-freq callback function signaled STOP then we need to cancel the async read
    if shared_buffer['stop-signal'] == True:
        sdr.cancel_read_async() # needs to be called from this function, so we use the shared memory to send a signal
    shared_buffer['utilization'] = (time.time() - startTime)/float(samples_per_batch)*sdr.sample_rate # should be below 1.0 to avoid overflows

# Function that runs asynchronous reading from the RTL, and is a blocking function
def start_sdr():
    while True:
        if shared_buffer['stop-signal'] == False:
            sdr.read_samples_async(callback=process_samples, num_samples=samples_per_batch) # sets RTL to asynchronous mode using the specified callback
        time.sleep(0.1) # we only wait here when stop-signal == True, which is usually very brief
        
# Start SDR sample processign as a separate thread
p = Process(target=start_sdr) 
p.start()


# Frequncy Sink (line plot)
fft_plot = pysdr.base_plot('Freq [MHz]', 'PSD [dB]', 'Frequency Sink', disable_horizontal_zooming=True) 
f = (np.linspace(-sdr.sample_rate/2.0, sdr.sample_rate/2.0, fft_size) + sdr.center_freq)/1e6
fft_line = fft_plot.line(f, np.zeros(len(f)), color="aqua", line_width=1) # set x values but use dummy values for y

# Time Sink (line plot)
time_plot = pysdr.base_plot('Time [ms]', ' ', 'Time Sink', disable_horizontal_zooming=True) 
t = np.linspace(0.0, samples_in_time_plots / sdr.sample_rate, samples_in_time_plots) * 1e3 # in ms
timeI_line = time_plot.line(t, np.zeros(len(t)), color="aqua", line_width=1) # set x values but use dummy values for y
timeQ_line = time_plot.line(t, np.zeros(len(t)), color="red", line_width=1) # set x values but use dummy values for y

# Waterfall Sink ("image" plot)
waterfall_plot = pysdr.base_plot(' ', 'Time', 'Waterfall', disable_all_zooming=True) 
waterfall_plot._set_x_range(0, fft_size) # Bokeh tries to automatically figure out range, but in this case we need to specify it
waterfall_plot._set_y_range(0, waterfall_samples)
waterfall_plot.axis.visible = False # i couldn't figure out how to update x axis when freq changes, so just hide them for now
waterfall_data = waterfall_plot.image(image = [shared_buffer['waterfall']],  # input has to be in list form
                                      x = 0, # start of x
                                      y = 0, # start of y
                                      dw = fft_size, # size of x
                                      dh = waterfall_samples, # size of y
                                      palette = "Spectral9") # closest thing to matlab's jet    

# IQ/Constellation Sink ("circle" plot)
iq_plot = pysdr.base_plot(' ', ' ', 'IQ Plot')
iq_plot._set_x_range(-1.0, 1.0) # this is to keep it fixed at -1 to 1. you can also just zoom out with mouse wheel and it will stop auto-ranging
iq_plot._set_y_range(-1.0, 1.0)
iq_data = iq_plot.circle(np.zeros(samples_in_time_plots), 
                         np.zeros(samples_in_time_plots),
                         line_alpha=0.0, # setting line_width=0 didn't make it go away, but this works
                         fill_color="aqua",
                         fill_alpha=0.5, 
                         size=4) # size of circles

# Utilization bar (standard plot defined in gui.py)
utilization_plot = pysdr.utilization_bar(1.0) # sets the top at 10% instead of 100% so we can see it move
utilization_data = utilization_plot.quad(top=[shared_buffer['utilization']], bottom=[0], left=[0], right=[1], color="#B3DE69") #adds 1 rectangle

def gain_callback(attr, old, new):
    shared_buffer['stop-signal'] = True # triggers a stop of the asynchronous read (cant change gain during it)
    time.sleep(0.5) # give time for the stop signal to trigger it- if you get a segfault then this needs to be increased
    sdr.gain = float(new) # set new gain
    shared_buffer['stop-signal'] = False  # turns off "stop" signal

def freq_callback(attr, old, new):
    shared_buffer['stop-signal'] = True # see above comments
    time.sleep(0.5)
    sdr.center_freq = float(new) # TextInput provides a string
    f = np.linspace(-sdr.sample_rate/2.0, sdr.sample_rate/2.0, fft_size) + sdr.center_freq
    fft_line.data_source.data['x'] = f/1e6 # update x axis of freq sink
    shared_buffer['stop-signal'] = False
    
# gain selector
gain_select = Select(title="Gain:", value=str(sdr.gain), options=[str(i/10.0) for i in sdr.get_gains()])
gain_select.on_change('value', gain_callback)

# center_freq TextInput
freq_input = TextInput(value=str(sdr.center_freq), title="Center Freq [Hz]")
freq_input.on_change('value', freq_callback)

# add the widgets to the document
curdoc().add_root(row([widgetbox(gain_select, freq_input), utilization_plot])) # widgetbox() makes them a bit tighter grouped than column()

# Add four plots to document, using the gridplot method of arranging them
curdoc().add_root(gridplot([[fft_plot, time_plot], [waterfall_plot, iq_plot]], sizing_mode="scale_width", merge_tools=False)) # Spacer(width=20, sizing_mode="fixed")


# This function gets called periodically, and is how the "real-time streaming mode" works   
def plot_update():  
    timeI_line.data_source.data['y'] = shared_buffer['i'] # send most recent I to time sink
    timeQ_line.data_source.data['y'] = shared_buffer['q'] # send most recent Q to time sink
    iq_data.data_source.data['x'] = shared_buffer['i'] # send most recent I to IQ
    iq_data.data_source.data['y'] = shared_buffer['q'] # send most recent Q to IQ
    fft_line.data_source.data['y'] = shared_buffer['psd'] # send most recent psd to freq sink
    waterfall_data.data_source.data['image'] = [shared_buffer['waterfall']] # send waterfall 2d array to waterfall sink
    utilization_data.data_source.data['top'] = [shared_buffer['utilization']] # send most recent utilization level (only need to adjust top of rectangle)

# Add a periodic callback to be run every x milliseconds
curdoc().add_periodic_callback(plot_update, 150) 

# pull out a theme from themes.py
#curdoc().theme = pysdr.black_and_white
             



    
    
