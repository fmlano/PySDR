#!/usr/bin/env python

import pysdr # our python package

import numpy as np
import time 
from scipy.signal import firwin # FIR filter design using the window method

from flask import Flask, render_template 

from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.embed import autoload_server
from bokeh.layouts import column, row, gridplot, Spacer, widgetbox
from bokeh.models import Select, TextInput
from bokeh.server.server import Server
from bokeh.util.browser import view # utility to Open a browser to view the specified location.

from uhd import libpyuhd

from multiprocessing import Process, Manager 

# Parameters
fft_size = 512               # output size of fft, the input size is the samples_per_batch
waterfall_samples = 100      # number of rows of the waterfall
samples_per_batch = 2044 #256*1024 # num of samples that we process at a time
samples_in_time_plots = 500  # should be less than samples_per_batch

center_freq = 101.1e6
samp_rate = 1e6
gain = 50


# Set up the shared buffer between threads (using multiprocessing's Manager).  it is global
manager = Manager()
shared_buffer = manager.dict() # there is also an option to use a list
shared_buffer['waterfall'] = np.ones((waterfall_samples, fft_size))*-100.0 # waterfall buffer
shared_buffer['psd'] = np.zeros(fft_size) # PSD buffer
shared_buffer['i'] = np.zeros(samples_in_time_plots) # I buffer (time domain)
shared_buffer['q'] = np.zeros(samples_in_time_plots) # Q buffer (time domain)
shared_buffer['stop-signal'] = False # used to signal RTL to stop (when it goes true)
shared_buffer['utilization'] = 0.0 # float between 0 and 1, used to store how the process_samples is keeping up
shared_buffer['usrp-signal'] = (False, '')

# create a streaming-type FIR filter (this should act the same as a FIR filter block in GNU Radio)
taps = firwin(numtaps=100, cutoff=200e3, nyq=samp_rate)
prefilter = pysdr.fir_filter(taps)



    
# Function that processes each batch of samples that comes in (currently, all DSP goes here)
def process_samples(samples):
    startTime = time.time()
    #samples = prefilter.filter(samples)
    PSD = 10.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples, fft_size)/float(fft_size)))**2) # calcs PSD
    waterfall = shared_buffer['waterfall'] # pull waterfall from buffer
    waterfall[:] = np.roll(waterfall, -1, axis=0) # shifts waterfall 1 row
    waterfall[-1,:] = PSD # fill last row with new fft results
    shared_buffer['waterfall'] = waterfall # you have to copy it back into the manager_list
    shared_buffer['psd'] = PSD # overwrites whatever was in psd buffer, so that the GUI uses the most recent one when it goes to refresh itself
    shared_buffer['i'] = np.real(samples[0:samples_in_time_plots]) # i buffer
    shared_buffer['q'] = np.imag(samples[0:samples_in_time_plots]) # q buffer
    shared_buffer['utilization'] = (time.time() - startTime)/float(samples_per_batch)*samp_rate # should be below 1.0 to avoid overflows

# Function that runs asynchronous reading from the RTL, and is a blocking function
def start_sdr():
    # Initialize USRP
    usrp = libpyuhd.usrp.multi_usrp('') 
    usrp.set_rx_rate(samp_rate, 0) # 2nd arg is channel
    usrp.set_rx_freq(libpyuhd.types.tune_request(center_freq), 0) # apparently you have to do the tune request function
    usrp.set_rx_gain(gain, 0)
    st_args = libpyuhd.usrp.stream_args("fc32", "sc16")
    st_args.channels = [0] # channel
    metadata = libpyuhd.types.rx_metadata()
    streamer = usrp.get_rx_stream(st_args)
    buffer_samps = streamer.get_max_num_samps()
    print "max_num_samps: ", buffer_samps
    recv_buffer = np.zeros(buffer_samps, dtype=np.complex64)
    recv_samps = 0 # keeps track of where we are in buffer
    stream_cmd = libpyuhd.types.stream_cmd(libpyuhd.types.stream_mode.start_cont)
    stream_cmd.stream_now = True
    streamer.issue_stream_cmd(stream_cmd)
    while True:
        if shared_buffer['usrp-signal'][0] == True: 
            eval('usrp.' + shared_buffer['usrp-signal'][1])
            shared_buffer['usrp-signal'] = (False, '')
        samps = streamer.recv(recv_buffer, metadata) # receive samples!
        if metadata.error_code != libpyuhd.types.rx_metadata_error_code.none:
            print(metadata.strerror())
        if samps:
            samples = recv_buffer[0:samps]
            process_samples(samples)
            #print samps
        
# Start SDR sample processign as a separate thread
p = Process(target=start_sdr) 
p.start()


# This is the Bokeh "document"
def main_doc(doc):
    # Frequncy Sink (line plot)
    fft_plot = pysdr.base_plot('Freq [MHz]', 'PSD [dB]', 'Frequency Sink', disable_horizontal_zooming=True) 
    f = (np.linspace(-samp_rate/2.0, samp_rate/2.0, fft_size) + center_freq)/1e6
    fft_line = fft_plot.line(f, np.zeros(len(f)), color="aqua", line_width=1) # set x values but use dummy values for y
    
    # Time Sink (line plot)
    time_plot = pysdr.base_plot('Time [ms]', ' ', 'Time Sink', disable_horizontal_zooming=True) 
    t = np.linspace(0.0, samples_in_time_plots / samp_rate, samples_in_time_plots) * 1e3 # in ms
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
        gain = new # set new gain (leave it as a string)
        print "Setting gain to ", gain
        command = 'set_rx_gain(' + gain + ', 0)'
        shared_buffer['usrp-signal'] = (True, command)

    def freq_callback(attr, old, new):
        center_freq = float(new) # TextInput provides a string
        f = np.linspace(-samp_rate/2.0, samp_rate/2.0, fft_size) + center_freq
        fft_line.data_source.data['x'] = f/1e6 # update x axis of freq sink
        print "Setting freq to ", center_freq
        command = 'set_rx_freq(libpyuhd.types.tune_request(' + str(center_freq) + '), 0)'
        shared_buffer['usrp-signal'] = (True, command)        

        
    # gain selector
    gain_select = Select(title="Gain:", value=str(gain), options=[str(i*10) for i in range(8)])
    gain_select.on_change('value', gain_callback)
    
    # center_freq TextInput
    freq_input = TextInput(value=str(center_freq), title="Center Freq [Hz]")
    freq_input.on_change('value', freq_callback)
    
    # add the widgets to the document
    doc.add_root(row([widgetbox(gain_select, freq_input), utilization_plot])) # widgetbox() makes them a bit tighter grouped than column()

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
        utilization_data.data_source.data['top'] = [shared_buffer['utilization']] # send most recent utilization level (only need to adjust top of rectangle)

    # Add a periodic callback to be run every x milliseconds
    print "GOT TO PERIODIC CALLBACK"
    doc.add_periodic_callback(plot_update, 150) 
    
    # pull out a theme from themes.py
    doc.theme = pysdr.black_and_white
                 

flask_app = Flask('__main__') # use '__main__' because this script is the top level

# GET routine (provides the html template)
@flask_app.route('/', methods=['GET'])  # going to http://localhost:5006 or whatever will trigger this route
def bkapp_page():
    script = autoload_server(url='http://localhost:5006/bkapp') # switch to server_document when pip uses new version of bokeh, autoload_server is being depreciated
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




    
    
