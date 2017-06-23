A python-based software-defined radio (SDR) framework designed for extremely rapid development of computationally intensive SDR apps, with an emphasis on utilization of mature 3rd party packages/libraries

Bokeh is used for the GUI, although you are free to use anything you want (e.g. matplotlib)

VOLK with python wrappers can be used as a faster alternative to numpy

All apps created with this framework are automatically web-based thanks to Flask

### Get started with:

1. Connect an RTL-SDR
2. `[sudo] pip install -r requirements.txt`
3. `python bokeh_rtl.py`
4. A web browser should pop up, showing the FM radio spectrum

### Install RTL-SDR Driver:

1. `git clone https://github.com/osmocom/rtl-sdr.git`
2. `mkdir build`, `cd build`
3. `cmake ../ -DINSTALL_UDEV_RULES=ON -DDETACH_KERNEL_DRIVER=ON`
4. `make`, `sudo make install`, `sudo ldconfig`
5. might need `sudo pip install pyrtlsdr`
