from __future__ import print_function # allows python3 print() to work in python2

from flask import Flask, render_template 

from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.embed import server_document
from bokeh.server.server import Server
from bokeh.util.browser import view # utility to Open a browser to view the specified location.

from multiprocessing import Process, Manager 

from pysdr.themes import black_and_white # removed relative imports to work with python3

# This is the equivalent of top block. in reality it's just doing the Bokeh and Flask stuff, not DSP
class pysdr_app:
    def __init__(self):
        print("creating new app")
        self.flask_app = Flask('__main__') # use '__main__' because this script is the top level

        # GET routine for root page
        @self.flask_app.route('/', methods=['GET'])  # going to http://localhost:5006 or whatever will trigger this route
        def bkapp_page():
            script = server_document(url='http://localhost:5006/bkapp')
            return render_template('index.html', script=script)
    
    def assemble_bokeh_doc(self, widgets, plots, plot_update, theme):
        def main_doc(doc):
            doc.add_root(widgets)  # add the widgets to the document
            doc.add_root(plots)  # Add four plots to document, using the gridplot method of arranging them
            doc.add_periodic_callback(plot_update, 150)  # Add a periodic callback to be run every x milliseconds
            doc.theme = theme
 
        # Create bokeh app
        self.bokeh_app = Application(FunctionHandler(main_doc)) # Application is "a factory for Document instances" and FunctionHandler "runs a function which modifies a document"
        
    def create_bokeh_server(self):
        self.io_loop = IOLoop.current() # creates an IOLoop for the current thread
        # Create the Bokeh server, which "instantiates Application instances as clients connect".  We tell it the bokeh app and the ioloop to use
        server = Server({'/bkapp': self.bokeh_app}, io_loop=self.io_loop, allow_websocket_origin=["localhost:8080"]) 
        server.start() # Start the Bokeh Server and its background tasks. non-blocking and does not affect the state of the IOLoop
        
    def create_web_server(self):
        # Create the web server using tornado (separate from Bokeh server)
        print('Opening Flask app with embedded Bokeh application on http://localhost:8080/')
        http_server = HTTPServer(WSGIContainer(self.flask_app)) # A non-blocking, single-threaded HTTP server. serves the WSGI app that flask provides. WSGI was created as a low-level interface between web servers and web applications or frameworks to promote common ground for portable web application development
        http_server.listen(8080) # this is the single-process version, there are multi-process ones as well
        # Open browser to main page
        self.io_loop.add_callback(view, "http://localhost:8080/") # calls the given callback (Opens browser to specified location) on the next I/O loop iteration. provides thread-safety
        
    def start_web_server(self):
        self.io_loop.start() # starts ioloop, and is blocking
