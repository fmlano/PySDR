import numpy as np
from scipy import signal
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import QSize, pyqtSlot
import pyqtgraph as pg

# Params
filename = 'example_signal.iq'

# Read in signal from file
x = np.fromfile(filename, dtype=np.uint8)
x = (x - 127.5)/256.0
x = x[1::2] + 1j * x[::2] # un-interleave the I and Q

#filename = 'slice_417759-517759.iq'
#x = np.fromfile(filename, dtype=np.complex128)

decimation = 100
fft_size = 2**14



class HelloWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(640, 480))    
        self.setWindowTitle("Hello world") 

        
        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   

        gridLayout = QGridLayout(self)     
        centralWidget.setLayout(gridLayout)
        
        
        horizontal_group = QtWidgets.QHBoxLayout() # this will be in the top row, containing multiple widgets
        gridLayout.addLayout(horizontal_group, 0, 0)

        title = QLabel("Hello World from PyQt", self) 
        title.setAlignment(QtCore.Qt.AlignCenter) 
        horizontal_group.addWidget(title)
        
        # Add the main plot
        p1 = pg.PlotWidget()
        p1.plot(np.real(x[::decimation]), pen=(255, 0, 0), name="I")
        p1.plot(np.imag(x[::decimation]), pen=(0, 255, 0), name="Q")
        gridLayout.addWidget(p1, 1, 0)
        

        
        # Add button to save selection to file
        self.button = QPushButton('PyQt5 button', self)
        self.button.clicked.connect(self.handleButton)
        horizontal_group.addWidget(self.button)
        
        # Add the region item to the main plot and create callback function
        lr = pg.LinearRegionItem([len(x)/decimation * 0.0, len(x)/decimation * 0.1])
        self.lo, self.hi = lr.getRegion()
        self.lo = int(self.lo)
        self.hi = int(self.hi)        
        def regionUpdated(regionItem):
            self.lo, self.hi = regionItem.getRegion()
            self.lo = int(self.lo * decimation)
            self.hi = int(self.hi * decimation)
            self.label.setText('selected range: ' + str(self.lo) + ' - ' + str(self.hi))
        lr.sigRegionChanged.connect(regionUpdated)
        p1.addItem(lr)    
            
        # Create dynamic label at the top
        self.label = QLabel(self)
        horizontal_group.addWidget(self.label)                
        self.label.setText('selected range: ' + str(self.lo) + ' - ' + str(self.hi)) # default text
        
        
        # Add the FFT plot
        self.p2 = pg.PlotWidget()
        gridLayout.addWidget(self.p2, 2, 0)
        
        # Add the FFT selectable InfiniteLine
        selection_line = pg.InfiniteLine(0.0, movable=True)
        def position_changed(line):
            pos = line.value()
            pos = int(pos * decimation)
            x_sub = x[pos:pos+fft_size]
            X_sub = 10.0*np.log10(np.fft.fftshift(np.abs(np.fft.fft(x_sub))))
            self.p2.clear()
            #self.p2.canvas.draw()
            self.p2.plot(X_sub)
        selection_line.sigPositionChanged.connect(position_changed)
        p1.addItem(selection_line)      
                    
        
    def handleButton(self):
        new_filename = "slice_" + str(self.lo) + "-" + str(self.hi) + ".iq"
        x[self.lo:self.hi].tofile(new_filename)
        self.label.setText(self.label.text() + ' saved!')
                
          
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = HelloWindow()
    mainWin.show()
    sys.exit( app.exec_() )
    
    
