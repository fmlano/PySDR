''' 
usage:
  specify iq file (e.g. made with file_sink in gnuradio) in the parameters below
  bokeh serve such_pysdr.py

open in your browser:
    http://localhost:5006/such_pysdr
'''

import numpy as np
import os
import pysdr # our python package

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox, gridplot
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure

# Parameters
file_name = 'example_signal.iq'
waterfall_fft_size = 128
waterfall_samples = 100 # rows in waterfall
default_samples = waterfall_fft_size * waterfall_samples # number of samples for time plots and for the waterfall
fft_size = 1024 # num samples used for fft, should always be less than default_samples
samp_rate = 10e6
center_freq = 2.4e9
time_plot_decimation = 10

num_samples = os.path.getsize(file_name)/8
f = open(file_name, 'rb') # read, binary
f.seek(0, os.SEEK_SET)
y = np.fromfile(f, dtype=np.complex64, count=default_samples)

def psd(x, N):
    return 10.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x, N)/float(N)))**2)

def waterfall(x):
    out = np.zeros((waterfall_samples, waterfall_fft_size))
    for i in range(waterfall_samples):
        out[i,:] = psd(x[i*waterfall_fft_size:(i+1)*waterfall_fft_size], waterfall_fft_size)
    np.clip(out, -60.0, -10.0, out)
    return out

# Frequncy Sink (line plot)
fft_plot = pysdr.base_plot('Freq [MHz]', 'PSD [dB]', 'Frequency Sink', disable_horizontal_zooming=True, plot_height=200) 
fft_x = (np.linspace(-samp_rate/2.0, samp_rate/2.0, fft_size) + center_freq)/1e6
source_fft = ColumnDataSource(data=dict(x=fft_x, y=psd(y, fft_size)))
fft_line = fft_plot.line('x', 'y', source=source_fft, color="aqua", line_width=1) # set x values but use dummy values for y


# Time Sink (line plot)
time_plot = pysdr.base_plot('Time [ms]', ' ', 'Time Sink', disable_horizontal_zooming=True, plot_height=200) 
t = np.linspace(0.0, default_samples / samp_rate, default_samples/time_plot_decimation) * 1e3 # in ms
source_i = ColumnDataSource(data=dict(x=t, y=np.real(y[0::time_plot_decimation])))
source_q = ColumnDataSource(data=dict(x=t, y=np.imag(y[0::time_plot_decimation])))
timeI_line = time_plot.line('x', 'y', source=source_i, color="aqua", line_width=1) # set x values but use dummy values for y
timeQ_line = time_plot.line('x', 'y', source=source_q, color="red", line_width=1) # set x values but use dummy values for y

# Waterfall Sink ("image" plot)
waterfall_plot = pysdr.base_plot(' ', 'Time', 'Waterfall', disable_all_zooming=True, plot_height=200) 
waterfall_plot._set_x_range(0, waterfall_fft_size) # Bokeh tries to automatically figure out range, but in this case we need to specify it
waterfall_plot._set_y_range(0, waterfall_samples)
waterfall_plot.axis.visible = False # i couldn't figure out how to update x axis when freq changes, so just hide them for now
waterfall_image = np.ones((waterfall_samples, waterfall_fft_size))*-100.0 # waterfall buffer, has to be in list form
source_waterfall = ColumnDataSource(data=dict(image=[waterfall_image]))
waterfall_data = waterfall_plot.image(image = 'image',  # input 
                                      x = 0, # start of x
                                      y = 0, # start of y
                                      dw = waterfall_fft_size, # size of x
                                      dh = waterfall_samples, # size of y
                                      source = source_waterfall, 
                                      palette = "Spectral9") # closest thing to matlab's jet  


# IQ/Constellation Sink ("circle" plot)
iq_plot = pysdr.base_plot(' ', ' ', 'IQ Plot')
#iq_plot._set_x_range(-1.0, 1.0) # this is to keep it fixed at -1 to 1. you can also just zoom out with mouse wheel and it will stop auto-ranging
#iq_plot._set_y_range(-1.0, 1.0)
source_iq = ColumnDataSource(data=dict(x=np.real(y[0::time_plot_decimation]), y=np.imag(y[0::time_plot_decimation])))
iq_data = iq_plot.circle('x', 'y', source=source_iq,
                         line_alpha=0.0, # setting line_width=0 didn't make it go away, but this works
                         fill_color="aqua",
                         fill_alpha=0.5, 
                         size=4) # size of circles
                         
                         
# Set up widgets
offset = Slider(title="offset", value=0, start=0, end=(num_samples-default_samples), step=128) # arbitrary step
amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)


def update_data(attrname, old, new):
    #a = amplitude.value

    # update plots
    f.seek(offset.value*8, os.SEEK_SET)
    y = np.fromfile(f, dtype=np.complex64, count=default_samples)
    t = np.linspace(0.0, default_samples / samp_rate, default_samples/time_plot_decimation) * 1e3 # in ms 
    source_i.data = dict(x=t, y=np.real(y[0::time_plot_decimation]))
    source_q.data = dict(x=t, y=np.imag(y[0::time_plot_decimation]))
    source_iq.data = dict(x=np.real(y[0::time_plot_decimation]), y=np.imag(y[0::time_plot_decimation]))
    source_fft.data = dict(x=fft_x, y=psd(y, fft_size))
    source_waterfall.data['image'] = [waterfall(y)]
     
     
for w in [offset, amplitude]:
    w.on_change('value', update_data)


# Set up layouts and add to document
widgets = row([widgetbox(offset, amplitude)]) # widgetbox() makes them a bit tighter grouped than column()
plots = gridplot([[fft_plot, time_plot], [waterfall_plot, iq_plot]], sizing_mode="scale_width")

curdoc().add_root(widgets)
curdoc().add_root(plots)
curdoc().title = "such pysdr"



