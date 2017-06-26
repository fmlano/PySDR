#!/usr/bin/env python

# to run this app use: bokeh serve such_samples_clone_nonflask.py --show

import pysdr # our python package

import numpy as np
import time 
import os

from bokeh.io import curdoc
from bokeh.application import Application
from bokeh.layouts import column, row, gridplot, Spacer, widgetbox
from bokeh.models import Select, TextInput, Slider

# Parameters
iq_file = '/home/marc/fm-recording.iq'  # for testing I created one using standard file sink in gnuradio
sample_rate = 10e6  # what did you use for recording it?
center_freq = 100e6 # what did you use for recording it?
current_window_size = 50000 # starting window size
fft_size = 128      # output size of fft, the input size is the samples_per_batch
decimation_factor = 500 # amount to decimate so we aren't displaying millions of points on a plot (also acts as the limit)

# Globals
current_start_sample = 0 # this can be changed with the slider
total_samples = os.path.getsize(iq_file)/8
waterfall_samples = int(current_window_size/fft_size) # number of rows of the waterfall


# Frequncy Sink (line plot)
fft_plot = pysdr.base_plot('Freq [MHz]', 'PSD [dB]', 'Frequency Sink', disable_horizontal_zooming=True) 
f = (np.linspace(-sample_rate/2.0, sample_rate/2.0, fft_size) + center_freq)/1e6
fft_line = fft_plot.line(f, np.zeros(len(f)), color="aqua", line_width=1) # set x values but use dummy values for y

# Time Sink (line plot)
time_plot = pysdr.base_plot('Time [ms]', 'Amplitude', 'Time Sink', disable_horizontal_zooming=True) 
decimation = int(current_window_size/decimation_factor) # so we only display decimation_factor points at most
if decimation > 1:
    t = np.linspace(0.0, current_window_size / sample_rate, current_window_size/decimation) * 1e3 # in ms
else:
    t = np.linspace(0.0, current_window_size / sample_rate, current_window_size) * 1e3 # in ms
timeI_line = time_plot.line(t, np.zeros(len(t)), color="aqua", line_width=1) # set x values but use dummy values for y
timeQ_line = time_plot.line(t, np.zeros(len(t)), color="red", line_width=1) # set x values but use dummy values for y

# Waterfall Sink ("image" plot)
waterfall_plot = pysdr.base_plot('Freq [MHz]', 'Time [ms]', 'Waterfall', plot_height=400, disable_all_zooming=True) 
waterfall_plot._set_x_range(f[0], f[-1]) 
waterfall_plot._set_y_range(0, t[-1])
waterfall_plot.yaxis.visible = False # i couldn't figure out how to update y axis when time scale changes, so just hide for now
waterfall_data = waterfall_plot.image(image = [np.zeros((waterfall_samples, fft_size))],  # input has to be in list form
                                      x = f[0], # start of x
                                      y = 0, # start of y
                                      dw = f[-1] - f[0], # size of x
                                      dh = t[-1], # size of y
                                      palette = "Spectral9") # closest thing to matlab's jet    

def process_samples():
    # read a portion of the file
    with open(iq_file) as fin:
        fin.seek(current_start_sample*8) # 8 bytes per complex sample
        samples = np.fromfile(fin, dtype=np.complex64, count = current_window_size)
    
    # DSP, and store results to buffers
    PSD = 10.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples, fft_size)/float(fft_size)))**2) # calcs PSD
    decimation = int(current_window_size/decimation_factor)
    if decimation > 1:
        i = np.real(samples[::decimation]) 
        q = np.imag(samples[::decimation]) 
    else:
        i = np.real(samples) 
        q = np.imag(samples) 

    # calc waterfall
    waterfall = np.zeros((waterfall_samples, fft_size))
    for waterfall_row in range(waterfall_samples):
        waterfall[waterfall_row,:] = 10.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples[waterfall_row*fft_size:(waterfall_row+1)*fft_size])/float(fft_size)))**2)
    # send results to GUI
    timeI_line.data_source.data['y'] = i # send most recent I to time sink
    timeQ_line.data_source.data['y'] = q # send most recent Q to time sink
    fft_line.data_source.data['y'] = PSD # send most recent psd to freq sink
    waterfall_data.data_source.data['image'] = [waterfall] # send waterfall 2d array to waterfall sink

def position_callback(attr, old, new):
    global current_start_sample # any variable we plan to change needs to be declared global
    current_start_sample = new
    process_samples() # reprocess samples

def window_size_callback(attr, old, new):
    global current_window_size
    global waterfall_samples
    current_window_size = int(new) # TextInput provides a string
    waterfall_samples = int(current_window_size/fft_size) # number of rows of the waterfall
    # update x axis of time sink and waterfall
    decimation = int(current_window_size/decimation_factor) 
    if decimation > 1:
        t = np.linspace(0.0, current_window_size / sample_rate, current_window_size/decimation) * 1e3 # in ms
    else:
        t = np.linspace(0.0, current_window_size / sample_rate, current_window_size) * 1e3 # in ms
    timeI_line.data_source.data['x'] = t
    timeQ_line.data_source.data['x'] = t
    process_samples() # reprocess since we changed the window size

# position slider
position_slider = Slider(start=0, end=(total_samples-current_window_size), value=0, step=100, title="Start Sample") #FIXME end is not correct
position_slider.on_change('value', position_callback)

# center_freq TextInput
window_size_input = TextInput(value=str(current_window_size), title="Window Size") # FIXME add limits
window_size_input.on_change('value', window_size_callback)

# add the widgets to the document
curdoc().add_root(widgetbox(position_slider, window_size_input)) # widgetbox() makes them a bit tighter grouped than column()

# assemble the plots the way we like to, and add to document
col1 = column([waterfall_plot], sizing_mode="scale_width")
col2 = column([time_plot, fft_plot], sizing_mode="scale_width")
row1 = row([col1, col2], sizing_mode="scale_width")
curdoc().add_root(row1)

# pull out a theme from themes.py
#curdoc().theme = pysdr.black_and_white

print "Starting!"
process_samples() # go ahead and process samples using the initial parameters (starting at sample 0)



             


    
