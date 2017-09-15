A "guide" for using Python as a software-defined radio (SDR) framework, with emphasis on extremely rapid development of SDR apps.  This can be achieved through utilization of mature 3rd party packages/libraries.  High sample rate performance is achieved via efficient hardware drivers (e.g. pysdruhd) and functions (e.g. using NumPy which relies on BLAS and LAPACK for efficient linear algebra computations and utilizes SIMD for vector operations).  

[Bokeh](http://bokeh.pydata.org/en/latest/) is used for the GUI, although you are free to use anything you want (e.g. [matplotlib](https://matplotlib.org/))

[VOLK](http://libvolk.org) with python wrappers can be used as a faster alternative to numpy

SDR apps created using pysdr are automatically web-based thanks to [Bokeh](http://bokeh.pydata.org/en/latest/) and [Flask](http://flask.pocoo.org/).  Designing your GUI is the same process as designing a web page, allowing for great flexibility. 

Getting started guide coming soon!
