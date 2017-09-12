#!/usr/bin/env python

from __future__ import print_function # allows python3 print() to work in python2

import pysdr # our python package
import pysdruhd as uhd # nathans amazing uhd wrapper
import numpy as np
import time 
from scipy.signal import firwin # FIR filter design using the window method
from bokeh.layouts import column, row, gridplot, Spacer, widgetbox
from bokeh.models import Select, TextInput
from multiprocessing import Process, Manager, Queue

##############
# Parameters #
##############
center_freq = 101.1e6
samp_rate = 12.5e6
gain = 50
fft_size = 512               # output size of fft, the input size is the samples_per_batch
waterfall_samples = 100      # number of rows of the waterfall
samples_in_time_plots = 500  # should be less than samples per batch (2044 for B200)

##############
# SET UP GUI #
##############

# Frequncy Sink (line plot)
fft_plot = pysdr.base_plot('Freq [MHz]', 'PSD [dB]', 'Frequency Sink', disable_horizontal_zooming=True) 
f = (np.linspace(-samp_rate/2.0, samp_rate/2.0, fft_size) + center_freq)/1e6
fft_plot._input_buffer['y'] = np.zeros(fft_size) # this buffer is how the DSP sends data to the plot in realtime
fft_line = fft_plot.line(f, np.zeros(fft_size), color="aqua", line_width=1) # set x values but use dummy values for y

# Time Sink (line plot)
time_plot = pysdr.base_plot('Time [ms]', ' ', 'Time Sink', disable_horizontal_zooming=True) 
t = np.linspace(0.0, samples_in_time_plots / samp_rate, samples_in_time_plots) * 1e3 # in ms
time_plot._input_buffer['i'] = np.zeros(samples_in_time_plots) # I buffer (time domain)
timeI_line = time_plot.line(t, np.zeros(len(t)), color="aqua", line_width=1) # set x values but use dummy values for y
time_plot._input_buffer['q'] = np.zeros(samples_in_time_plots) # Q buffer (time domain)
timeQ_line = time_plot.line(t, np.zeros(len(t)), color="red", line_width=1) # set x values but use dummy values for y

# Waterfall Sink ("image" plot)
waterfall_plot = pysdr.base_plot(' ', 'Time', 'Waterfall', disable_all_zooming=True) 
waterfall_plot._set_x_range(0, fft_size) # Bokeh tries to automatically figure out range, but in this case we need to specify it
waterfall_plot._set_y_range(0, waterfall_samples)
waterfall_plot.axis.visible = False # i couldn't figure out how to update x axis when freq changes, so just hide them for now
waterfall_plot._input_buffer['waterfall'] = [np.ones((waterfall_samples, fft_size))*-100.0] # waterfall buffer, has to be in list form
waterfall_data = waterfall_plot.image(image = waterfall_plot._input_buffer['waterfall'],  # input 
                                      x = 0, # start of x
                                      y = 0, # start of y
                                      dw = fft_size, # size of x
                                      dh = waterfall_samples, # size of y
                                      palette = "Spectral9") # closest thing to matlab's jet    

# IQ/Constellation Sink ("circle" plot)
iq_plot = pysdr.base_plot(' ', ' ', 'IQ Plot')
#iq_plot._set_x_range(-1.0, 1.0) # this is to keep it fixed at -1 to 1. you can also just zoom out with mouse wheel and it will stop auto-ranging
#iq_plot._set_y_range(-1.0, 1.0)
iq_plot._input_buffer['i'] = np.zeros(samples_in_time_plots) # I buffer (time domain)
iq_plot._input_buffer['q'] = np.zeros(samples_in_time_plots) # I buffer (time domain)
iq_data = iq_plot.circle(np.zeros(samples_in_time_plots), 
                         np.zeros(samples_in_time_plots),
                         line_alpha=0.0, # setting line_width=0 didn't make it go away, but this works
                         fill_color="aqua",
                         fill_alpha=0.5, 
                         size=4) # size of circles

# Utilization bar (standard plot defined in gui.py)
utilization_plot = pysdr.utilization_bar(1.0) # sets the top at 10% instead of 100% so we can see it move
utilization_plot._input_buffer['y'] = [0.0] # float between 0 and 1, used to store how the process_samples is keeping up
utilization_data = utilization_plot.quad(top=utilization_plot._input_buffer['y'], bottom=[0], left=[0], right=[1], color="#B3DE69") #adds 1 rectangle

# Queue used to send usrp commands from Bokeh thread to the USRP thread
usrp_command_queue = Queue() 

def gain_callback(attr, old, new):
    gain = new # set new gain (leave it as a string)
    print("Setting gain to ", gain)
    command = 'set_gain("A:A",' + gain + ')'
    usrp_command_queue.put(command)

def freq_callback(attr, old, new):
    center_freq = float(new) # TextInput provides a string
    f = np.linspace(-samp_rate/2.0, samp_rate/2.0, fft_size) + center_freq
    fft_line.data_source.data['x'] = f/1e6 # update x axis of freq sink
    print("Setting freq to ", center_freq)
    command = 'set_frequency("A:A",' + str(center_freq) + ')'
    usrp_command_queue.put(command)

# gain selector
gain_select = Select(title="Gain:", value=str(gain), options=[str(i*10) for i in range(8)])
gain_select.on_change('value', gain_callback)

# center_freq TextInput
freq_input = TextInput(value=str(center_freq), title="Center Freq [Hz]")
freq_input.on_change('value', freq_callback)

widgets = row([widgetbox(gain_select, freq_input), utilization_plot]) # widgetbox() makes them a bit tighter grouped than column()
plots = gridplot([[fft_plot, time_plot], [waterfall_plot, iq_plot]], sizing_mode="scale_width", ) # Spacer(width=20, sizing_mode="fixed")

# This function gets called periodically, and is how the "real-time streaming mode" works   
def plot_update():  
    timeI_line.data_source.data['y'] = time_plot._input_buffer['i'] # send most recent I to time sink
    timeQ_line.data_source.data['y'] = time_plot._input_buffer['q'] # send most recent Q to time sink
    iq_data.data_source.data = {'x': iq_plot._input_buffer['i'], 'y': iq_plot._input_buffer['q']} # send I and Q in one step using dict
    fft_line.data_source.data['y'] = fft_plot._input_buffer['y'] # send most recent psd to freq sink
    waterfall_data.data_source.data['image'] = waterfall_plot._input_buffer['waterfall'] # send waterfall 2d array to waterfall sink
    utilization_data.data_source.data['top'] = utilization_plot._input_buffer['y'] # send most recent utilization level (only need to adjust top of rectangle)


###################
# Init DSP Blocks #
###################

# create a streaming-type FIR filter (this should act the same as a FIR filter block in GNU Radio)
taps = firwin(numtaps=100, cutoff=200e3, nyq=samp_rate) # scipy's filter designer
prefilter = pysdr.fir_filter(taps)
accumulator = pysdr.accumulator(100000) # accumulates batches of samples so we can process more at a time. arg is min amount to store

###############
# DSP Routine #
###############

# Function that processes each batch of samples that comes in (currently, all DSP goes here)
def process_samples(samples):
    startTime = time.time()
    if accumulator.accumulate_samples(samples): # add samples to accumulator (returns True when we have enough)
        samples = accumulator.samples # messy way of doing it but it works
        #samples = prefilter.filter(samples) # uncomment this to add a filter
        PSD = 10.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples, fft_size)/float(fft_size)))**2) # calcs PSD
        # add row to waterfall
        waterfall = waterfall_plot._input_buffer['waterfall'][0] # pull waterfall from buffer
        waterfall[:] = np.roll(waterfall, -1, axis=0) # shifts waterfall 1 row
        waterfall[-1,:] = PSD # fill last row with new fft results
        # stick everything we want to display into the shared buffer
        waterfall_plot._input_buffer['waterfall'] = [waterfall] # remember to copy it back into the buffer
        fft_plot._input_buffer['y'] = PSD # overwrites whatever was in psd buffer, so that the GUI uses the most recent one when it goes to refresh itself
        time_plot._input_buffer['i'] = np.real(samples[0:samples_in_time_plots]) # i buffer
        time_plot._input_buffer['q'] = np.imag(samples[0:samples_in_time_plots]) # q buffer
        iq_plot._input_buffer['i'] = np.real(samples[0:samples_in_time_plots])
        iq_plot._input_buffer['q'] = np.imag(samples[0:samples_in_time_plots])
        utilization_plot._input_buffer['y'] = [(time.time() - startTime)/float(len(samples))*samp_rate] # should be below 1.0 to avoid overflows
        
###############
# USRP Config #
###############

def run_usrp():
    usrp = uhd.Usrp(streams={"A:A": {'antenna': 'RX2', 'frequency':center_freq, 'gain':60}}, rate=samp_rate) # need to use A:0 for x310
    usrp.send_stream_command({'now': True}) # start streaming
    ''' uncomment this to use Ettus' pyuhd
    usrp = pysdr.usrp_source('') # this is where you would choose which addr or usrp type
    usrp.set_samp_rate(samp_rate) 
    usrp.set_center_freq(center_freq)
    usrp.set_gain(gain)
    usrp.prepare_to_rx()
    '''
    while True: # endless loop of rx samples
        if not usrp_command_queue.empty():  # check if there's a usrp command in the queue
            command = usrp_command_queue.get()
            eval('usrp.' + command) # messy way to do it!
        samples, metadata = usrp.recv() # receive samples. pretty sure this function is blocking
        process_samples(samples) # send samples to DSP
        
# We do run_usrp() and process_samples() in a 2nd thread, while the Bokeh GUI stuff is in the main thread
usrp_dsp_process = Process(target=run_usrp) 
usrp_dsp_process.start()


################
# Assemble App #
################
myapp = pysdr.pysdr_app() # start new pysdr app
myapp.assemble_bokeh_doc(widgets, plots, plot_update, pysdr.black_and_white) # widgets, plots, periodic callback function, theme
myapp.create_bokeh_server()
myapp.create_web_server() 
myapp.start_web_server() # start web server.  blocking








