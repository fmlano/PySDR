#!/usr/bin/env python

import numpy as np
import yaml
import time 

from flask import Flask, render_template

from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.embed import autoload_server
from bokeh.layouts import column, row, gridplot, Spacer, widgetbox
from bokeh.models import Slider, Dropdown, Select, TextInput
from bokeh.models import WheelZoomTool, BoxZoomTool, ResetTool, SaveTool # all the tools we want- reference http://bokeh.pydata.org/en/0.10.0/docs/reference/models/tools.html
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from bokeh.util.browser import view # utility to Open a browser to view the specified location.

from rtlsdr import RtlSdr

from multiprocessing import Process, Manager 

# Parameters
fft_size = 512               # output size of fft, the input size is the samples_per_batch
waterfall_samples = 100      # number of rows of the waterfall
samples_per_batch = 256*1024 # num of samples that we process at a time

# RTL-SDR stuff (including how to install the driver and wrapper)
#    git clone https://github.com/osmocom/rtl-sdr.git
#    mkdir build, cd build
#    cmake ../ -DINSTALL_UDEV_RULES=ON -DDETACH_KERNEL_DRIVER=ON
#    make, sudo make install, sudo ldconfig
#    sudo pip install pyrtlsdr
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
shared_buffer['i'] = np.zeros(fft_size)   # I buffer (time domain)
shared_buffer['q'] = np.zeros(fft_size)   # Q buffer (time domain)
shared_buffer['stop-signal'] = False      # used to signal RTL to stop (when it goes true)

# Function that processes each batch of samples that comes in (currently, all DSP goes here)
def process_samples(samples, rtlsdr_obj):
    #startTime = time.time()
    PSD = 10.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples, fft_size)/float(fft_size)))**2) # calcs PSD
    waterfall = shared_buffer['waterfall'] # pull waterfall from buffer
    waterfall[:] = np.roll(waterfall, -1, axis=0) # shifts waterfall 1 row
    waterfall[-1,:] = PSD # fill last row with new fft results
    shared_buffer['waterfall'] = waterfall # you have to copy it back into the manager_list
    shared_buffer['psd'] = PSD # overwrites whatever was in psd buffer, so that the GUI uses the most recent one when it goes to refresh itself
    shared_buffer['i'] = np.real(samples[0:fft_size]) # i buffer
    shared_buffer['q'] = np.imag(samples[0:fft_size]) # q buffer
    # if the change-gain or change-freq callback function signaled STOP then we need to cancel the async read
    if shared_buffer['stop-signal'] == True:
        sdr.cancel_read_async() # needs to be called from this function, so we use the shared memory to send a signal
    #print 'Processing samples at ', samples_per_batch/(time.time() - startTime)/1e6, ' MHz' # this number should be greater than the rtl sample rate

# Function that runs asynchronous reading from the RTL, and is a blocking function
def start_sdr():
    while True:
        if shared_buffer['stop-signal'] == False:
            sdr.read_samples_async(callback=process_samples, num_samples=samples_per_batch) # sets RTL to asynchronous mode using the specified callback
        time.sleep(0.1) # we only wait here when stop-signal == True, which is usually very brief
        
# Start SDR sample processign as a separate thread
p = Process(target=start_sdr) 
p.start()

# This is the Bokeh "document"
def main_doc(doc):
    # Frequncy Sink (line plot)
    fft_plot = figure(plot_width=300, # this is more for the ratio, because we have auto-width scaling
                      plot_height=200,
                      y_range=(-70, -10), # dB 
                      y_axis_label='PSD [dB]',
                      x_axis_label='Freq [MHz]',
                      tools=[BoxZoomTool(), WheelZoomTool(dimensions='height'), ResetTool(), SaveTool()],
                      title="Frequency Sink")  # use min_border=30 to add padding between plots
    fft_plot.toolbar.active_scroll = fft_plot.toolbar.tools[1] # sets wheel zoom active by default (tools[1] is the wheelzoom)
    fft_plot.toolbar.logo=None # hides stupid bokeh logo
    fft_line = fft_plot.line(range(fft_size), range(fft_size), color="aqua", line_width=1) 
    fft_line.data_source.data['x'] = (np.linspace(-sdr.sample_rate/2.0, sdr.sample_rate/2.0, fft_size) + sdr.center_freq)/1e6 
    
    # Time Sink (line plot)
    time_plot = figure(plot_width=300, # this is more for the ratio, because we have auto-width scaling
                       plot_height=200,
                       y_range=(-1.0, 1.0), # amplitude 
                       y_axis_label=' ',
                       x_axis_label='Time [ms]',
                       tools=[BoxZoomTool(), WheelZoomTool(dimensions='height'), ResetTool(), SaveTool()],
                       title="Time Sink")
    time_plot.toolbar.active_scroll = time_plot.toolbar.tools[1] # sets wheel zoom active by default (tools[1] is the wheelzoom)
    time_plot.toolbar.logo=None # hides stupid bokeh logo
    samples_to_plot = fft_size # to make it easier
    timeI_line = time_plot.line(range(samples_to_plot), range(samples_to_plot), color="aqua", line_width=1) 
    timeQ_line = time_plot.line(range(samples_to_plot), range(samples_to_plot), color="red", line_width=1) 
    timeI_line.data_source.data['x'] = np.linspace(0.0, samples_to_plot / sdr.sample_rate, samples_to_plot) * 1e3 # in ms
    timeQ_line.data_source.data['x'] = timeI_line.data_source.data['x'] # same x for both lines

    # Waterfall Sink ("image" plot)
    waterfall_plot = figure(plot_width=300, # this is more for the ratio, because we have auto-width scaling
                            plot_height=200,
                            x_range=(0, fft_size), 
                            y_range=(0, waterfall_samples),
                            y_axis_label='Time',
                            x_axis_label='',
                            tools=[BoxZoomTool(), ResetTool(), SaveTool()],
                            title="Waterfall")
    waterfall_plot.toolbar.logo=None # hides stupid bokeh logo
    waterfall_plot.axis.visible = False # i couldn't figure out how to update x axis when freq changes, so just hide them for now
    waterfall_data = waterfall_plot.image(image=[np.ones((waterfall_samples, fft_size))*-100.0],  # input has to be in list form
                                          x=0, 
                                          y=0, 
                                          dw=fft_size, 
                                          dh=waterfall_samples, 
                                          palette="Spectral9") # closest thing to matlab's jet    

    # IQ/Constellation Sink ("circle" plot)
    iq_plot = figure(plot_width=300, # this is more for the ratio, because we have auto-width scaling
                     plot_height=200,
                     x_range=(-1.0, 1.0), # amplitude 
                     y_range=(-1.0, 1.0), # amplitude 
                     tools=[BoxZoomTool(), WheelZoomTool(), ResetTool(), SaveTool()],
                     title="IQ Plot")
    iq_plot.toolbar.active_scroll = iq_plot.toolbar.tools[1] # sets wheel zoom active by default (tools[1] is the wheelzoom)
    iq_plot.toolbar.logo=None # hides stupid bokeh logo
    samples_to_plot = fft_size # to make it easier
    iq_data = iq_plot.circle(range(samples_to_plot), 
                                range(samples_to_plot),
                                line_alpha=0.0, # setting line_width=0 didn't make it go away 
                                fill_color="aqua",
                                fill_alpha=0.5,
                                size=4)

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
    doc.add_root(widgetbox(gain_select, freq_input)) # widgetbox() makes them a bit tighter grouped than column()

    # Add four plots to document, using the gridplot method of arranging them
    doc.add_root(gridplot([[fft_plot, time_plot], [waterfall_plot, iq_plot]], sizing_mode="scale_width", merge_tools=False)) # Spacer(width=20, sizing_mode="fixed")
    
    # This function gets called periodically, and is how the "real-time streaming mode" works   
    def plot_update():  
        timeI_line.data_source.data['y'] = shared_buffer['i'] # send most recent I to time sink
        timeQ_line.data_source.data['y'] = shared_buffer['q'] # send most recent Q to time sink
        iq_data.data_source.data['x'] = shared_buffer['i'] # send most recent I to IQ
        iq_data.data_source.data['y'] = shared_buffer['q'] # send most recent Q to IQ
        fft_line.data_source.data['y'] = shared_buffer['psd'] # send most recent psd to freq sink
        waterfall_data.data_source.data['image'] = [shared_buffer['waterfall']] # send waterfall 2d array to waterfall sink

    # Add a periodic callback to be run every x milliseconds
    doc.add_periodic_callback(plot_update, 150) 
    
    # nice way of providing a theme for all the figures in the document (could go in a sep file eventually)
    doc.theme = Theme(json=yaml.load("""
        attrs:
            Figure:
                background_fill_color: "#333333"
                outline_line_color: white
                toolbar_location: right
                border_fill_color: black
            Axis:
                axis_label_text_color: "white"
                major_label_text_color: "white"
                axis_label_text_font_style: bold
                major_tick_line_color: "white"
                minor_tick_line_color: "white"
            Title:
                text_color: "white"
                text_font_style: bold
            Grid:
                grid_line_dash: [2, 2]
                grid_line_color: gray """))
                 

flask_app = Flask('__main__') # use '__main__' because this script is the top level

# GET routine (provides the html template)
@flask_app.route('/', methods=['GET'])  # going to http://localhost:5006 or whatever will trigger this route
def bkapp_page():
    script = autoload_server(url='http://localhost:5006/bkapp')
    return render_template('index.html', script=script)
    
if __name__ == '__main__':
    # Create bokeh app and IOLoop
    bokeh_app = Application(FunctionHandler(main_doc)) # Application is "a factory for Document instances" and FunctionHandler "runs a function which modifies a document"
    io_loop = IOLoop.current() # creates an IOLoop for the current thread
    # Create the Bokeh server, which "instantiates Application instances as clients connect".  We tell it the bokeh app and the ioloop to use
    server = Server({'/bkapp': bokeh_app}, io_loop=io_loop, allow_websocket_origin=["localhost:8080"]) 
    server.start() # Start the Bokeh Server and its background tasks. non-blocking and does not affect the state of the IOLoop
    # Create the web server using tornado (separate from Bokeh server)
    print('Opening Flask app with embedded Bokeh application on http://localhost:8080/')
    http_server = HTTPServer(WSGIContainer(flask_app)) # A non-blocking, single-threaded HTTP server. serves the WSGI app that flask provides. WSGI was created as a low-level interface between web servers and web applications or frameworks to promote common ground for portable web application development
    http_server.listen(8080) # this is the single-process version, there are multi-process ones as well
    # Open browser to main page
    io_loop.add_callback(view, "http://localhost:8080/") # calls the given callback (Opens browser to specified location) on the next I/O loop iteration. provides thread-safety
    io_loop.start() # starts ioloop, and is blocking




    
    
