import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import *
from pylab import *
import time
from scipy import signal

# Configure RTL-SDR parameters
sdr = RtlSdr()
sdr.sample_rate = 2.4e6  # 2.4 MHz
sdr.center_freq = 160270968 # MHz
sdr.gain = 'auto'

# Number of samples to read in each iteration
samples_to_process = 1.e5
threshold = 0.6
freq_offset = -43.8e4 #Hz

lastfalling = 0
last = 0
lastrise = 0
index = 0
time_between_rising_edge = 0
rssi = 0
pulse_width = 0

def process_samples(samples, sample_rate, freq_offset, threshold):

    global index
    global lastrise
    global last
    global lastfalling
    global time_between_rising_edge
    global rssi
    global pulse_width

    t = np.arange(len(samples))/sample_rate
    samples = samples * np.exp(2j*np.pi*t*freq_offset)
    h = signal.firwin(501, 0.02, pass_zero=True)
    samples = np.convolve(samples, h, 'valid')
    samples = samples[::100]
    sample_rate = sample_rate/100
    samples = np.abs(samples)
    samples = np.convolve(samples, [1]*10, 'valid')/10
    max_samp = np.max(samples)
    samples /= np.max(samples)

    # plt.plot(samples)
    # plt.show()

    def islow(v):
        return v < threshold
    def ishigh(v):
        return v >= threshold

    if np.max(samples) < threshold:
        return
    
    # samples array index(j), value(n) 
    for j,n in enumerate(samples):
        i = index + j
        # i = j        
        # Is this a rising edge?
        if ishigh(n) and islow(last):
            # then check is second rising edge? i.e. lastrise!=0
            if lastrise:
                time_between_rising_edge = round(sample_rate/(i-lastrise)*60,2)
                print(f"BPM: {time_between_rising_edge}")
            lastrise = i
        # Is this a falling edge?
        elif islow(n) and ishigh(last):
            pulse_width =  round((i-lastrise)/sample_rate,4)
            lastfalling = i      
            rssi = round(np.mean(samples[j-(i-lastrise):j]) * max_samp,2)
            print(f"rssi : {rssi}")
        last = n
    index = index + len(samples)
                
try:
    while True:
        # Read a chunk of samples
        # samples_to_process is number of samples to process
        samples = sdr.read_samples(samples_to_process)

        # # use matplotlib to estimate and plot the PSD
        # psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
        # xlabel('Frequency (MHz)')
        # ylabel('Relative power (dB)')
        # show()

        # Process the samples
        processed_data = process_samples(samples, sdr.sample_rate, freq_offset, threshold)

except KeyboardInterrupt:
    # Stop the loop on keyboard interrupt
    print("Program terminated.")

finally:
    # Close the RTL-SDR device
    sdr.close()

