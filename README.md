## Summary

A "guide" for using Python as a software-defined radio (SDR) framework, with emphasis on extremely rapid development of SDR apps. 
This can be achieved through utilization of mature 3rd party packages/libraries.
High sample rate performance is achieved via efficient hardware drivers and functions
(e.g. using NumPy which relies on BLAS and LAPACK for efficient linear algebra computations and utilizes SIMD for vector operations).

PyQt5, including PyQtGraph, is used for creating pretty and high-refresh-rate GUIs.

[VOLK](http://libvolk.org) with python wrappers can be used as a faster alternative to numpy.

## Install Guide

As of late 2018, UHD's Python API is too new to be included in the UHD packaged with Ubuntu, so we will be installing it from source.  Hopefully it will get packaged soon, which will make the steps below way shorter.

This guide requires using Ubuntu 18, mainly because the Boost packaged with Ubuntu 16 and 14 is several years old.  In order to use these older distros you probably have to install Boost from source.

Before we start it's best you don't have UHD installed at the system level, so do `sudo apt-get remove libuhd*` and `sudo apt-get remove uhd-host`

We are going to install UHD, PyQT5, pyqtgraph, and all prereqs in an isolate python virtual evironment (not to be confused with a VM), which will let us blow away installs by just removing one directory, and make sure we don't mess up anything else installed on this system.

- `cd ~`
- `sudo pip3 install virtualenv` (create the python virtual environment)
- `virtualenv -p python3 python_uhd_install` (makes sure python3 is used)
- `cd python_uhd_install`
- `echo "export LD_LIBRARY_PATH=\$VIRTUAL_ENV/lib/" >> bin/activate` (adds lib to path)
- `. bin/activate`
- `pip install numpy mako requests six pyqt5 pyqtgraph`
- `mkdir src`
- `cd src`
- `git clone https://github.com/EttusResearch/uhd.git` (we will be using master)
- `cd uhd/host`
- `mkdir build`
- `cd build`
- The following cmake command forces use of python3, disabled a bunch of UHD components we dont need (otherwise it takes ages), and installs UHD to lib within our virtualenv directory
- `cmake -DENABLE_PYTHON3=ON -DENABLE_EXAMPLES=OFF -DENABLE_TESTS=OFF -DENABLE_RFNOC=OFF -DENABLE_C_API=OFF -DENABLE_X300=OFF -DENABLE_B100=OFF -DENABLE_USRP1=OFF -DENABLE_USRP2=OFF -DENABLE_N230=OFF -DENABLE_N300=OFF -DENABLE_E320=OFF -DENABLE_OCTOCLOCK=OFF -DENABLE_MANUAL=OFF -DCMAKE_INSTALL_PREFIX=~/python_uhd_install/ ..`
- The output of the above command should include "LibUHD - Python API", "USB", "Utils", and "B200" in the list of enabled components.  If not, don't move on
- `make -j8` (build with 8 threads)
- `make install` (no sudo!)
- Plug in your B200/B210/B200mini
- `~/python_uhd_install/lib/uhd/utils/uhd_images_downloader.py`
- `uhd_find_devices`
- Hopefully it shows up. If it yells about permissions try `sudo apt-get install libusb-1.0-0-dev`
- `uhd_usrp_probe` (gives more info about your USRP)
- At this point we will clone pysdr and run a test app
- `cd ~` (you can clone pysdr wherever you want)
- `git clone https://github.com/pysdr/pysdr.git`
- `cd pysdr`
- `python usrp_qt5_app.py`
- A window should pop up, and you should see the spectrum of the FM band.  What displays is very similar to `uhd_fft` if you have ever used that.

You may or may not need to:
`sudo apt-get install libboost-all-dev`
