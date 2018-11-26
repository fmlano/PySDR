import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

filename = 'keyfob.iq'


x = np.fromfile(filename, dtype=np.uint8)

x = (x - 127.5)/256.0

x = x[1::2] + 1j * x[::2] # un-interleave the I and Q

x = x[466700:737290] # clip where i know the signal starts and stops

x = x[50000:] # remove some more so i know the signal is active


# Designing a filter to filter out negative (and another one for positive) frequencies:

H1 = [0]*128 + [1]*128
h1 = np.fft.ifftshift(np.fft.ifft(H1)) * np.hamming(256) # this saves the negative freqs
h2 = np.conj(h1) # this saves the positive freqs

x1 = signal.lfilter(h1, 1, x)
x2 = signal.lfilter(h2, 1, x)

# envelope detector filter (abs then lpf)
h3 = signal.firwin(33, 0.02)

x1 = signal.lfilter(h3, 1, np.abs(x1))
x2 = signal.lfilter(h3, 1, np.abs(x2))

#X = np.log10(np.fft.fftshift(np.abs(np.fft.fft(x, 4096))))

plt.plot(x1[::10])
plt.plot(x2[::10])
plt.show()





