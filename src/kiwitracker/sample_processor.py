from __future__ import annotations
from typing import TYPE_CHECKING
import asyncio
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt
from scipy import signal
import time
import math
from kiwitracker.beep_state_machine import BeepStateMachine
from kiwitracker.common import SamplesT, FloatArray, ProcessConfig
from datetime import datetime

def snr(samples, rising_edge_idx, falling_edge_idx, beep_slice):
    #print(f"rising edge in snr : {rising_edge_idx}")
    if (beep_slice):
        #print("here")
        noise_pwr = np.var( samples[falling_edge_idx:] )
        signal_pwr = np.var ( samples[0:falling_edge_idx])
    else:
        noise_pwr = np.var( np.concatenate([samples[:rising_edge_idx], samples[falling_edge_idx:]]) )
        signal_pwr = np.var ( samples[rising_edge_idx:falling_edge_idx] )
        #print(f" noise_pwr : {noise_pwr}")
        #print(f" signal_pwr : {signal_pwr}")
    snr_db = 10 * np.log10 ( signal_pwr / noise_pwr )
    return snr_db

class SampleProcessor:
    config: ProcessConfig
    threshold: float = 0.2
    beep_duration: float = 0.017 # seconds
    stateful_index: int
    beep_slice: bool
    rising_edge: int
    falling_edge: int

    def __init__(self, config: ProcessConfig) -> None:
        self.config = config
        self.stateful_index = 0
        self._time_array = None
        self._fir = None
        self._phasor = None
        self.stateful_rising_edge = 0
        self.beep_slice = False
        self.rising_edge = 0
        self.falling_edge = 0
        self.bsm = BeepStateMachine(config)

    @property
    def channel(self): 
        """Channel Number from Freq"""
        return math.floor((self.config.carrier_freq - 160.11e6)/0.01e6)

    @property
    def sample_rate(self): return self.config.sample_config.sample_rate

    @property
    def num_samples_to_process(self): return self.config.num_samples_to_process

    @property
    def carrier_freq(self):
        """Center frequency of the carrier wave to process (in Hz)"""
        return self.config.carrier_freq

    @property
    def freq_offset(self) -> float:
        """The offset (difference) between the sdr's center frequency and
        the :attr:`carrier_freq`
        """
        fc = self.config.sample_config.center_freq
        return fc - self.carrier_freq

    @property
    def time_array(self) -> FloatArray:
        t = self._time_array
        if t is None:
            t = np.arange(self.num_samples_to_process)/self.sample_rate
            self._time_array = t
        return t

    @property
    def fir(self) -> FloatArray:
        h = self._fir
        if h is None:
            h = self._fir = signal.firwin(501, 0.02, pass_zero=True)
        return h

    @property
    def phasor(self) -> npt.NDArray[np.complex128]:
        p = self._phasor
        if p is None:
            t = self.time_array
            p = np.exp(2j*np.pi*t*self.freq_offset)
            self._phasor = p
        return p

    @property
    def fft_size(self) -> int:
        # this makes sure there's at least 1 full chunk within each beep
        return int(self.beep_duration * self.sample_rate / 2)

    def process(self, samples: SamplesT):
        # look for the presence of a beep within the chunk and :
        # (1) if beep found calculate the offset
        # (2) if beep not found iterate the counters and move on

        fft_size = self.fft_size
        f = np.linspace(self.sample_rate/-2, self.sample_rate/2, fft_size)


        size = fft_size # 8704
        # step = self.sample_rate * self.beep_duration + 1000 
        step = 6704
        samples_to_send_to_fft = [samples[i : i + size] for i in range(0, len(samples), step)]

        num_ffts = len(samples_to_send_to_fft) # // is an integer division which rounds down
        fft_thresh = 0.1
        beep_freqs = []

        i = 0 
        for i in range(num_ffts):

            # fft = np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size]))) / fft_size
            fft = np.abs(np.fft.fftshift(np.fft.fft(samples_to_send_to_fft[i]))) / fft_size
            if np.max(fft) > fft_thresh:
                beep_freqs.append(np.linspace(self.sample_rate/-2, self.sample_rate/2, fft_size)[np.argmax(fft)])
                # beep_freqs.append(self.sample_rate/-2+np.argmax(fft)/fft_size*self.sample_rate) more efficent??


        # if no beeps increment and exit early
        if len(beep_freqs) == 0:
            self.stateful_index += (samples.size/100)
            return
        
        t = self.time_array
        samples = samples * self.phasor
        # next two lines are band pass filter?
        h = self.fir
        samples = signal.convolve(samples, h, 'valid')
        # decimation
        samples = samples[::100]
        # recalculation of sample rate due to decimation
        sample_rate = self.sample_rate/100
        samples_for_snr = samples
        samples = np.abs(samples)

        # smoothing
        samples = signal.convolve(samples, [1]*10, 'valid')/10
        #plt.plot(samples)
        #plt.show()
        # max_samp = np.max(samples)

        # Get a boolean array for all samples higher or lower than the threshold
        low_samples = samples < self.threshold
        high_samples = samples >= self.threshold

        # Compute the rising edge and falling edges by comparing the current value to the next with
        # the boolean operator & (if both are true the result is true) and converting this to an index
        # into the current array
        rising_edge_idx = np.nonzero(low_samples[:-1] & np.roll(high_samples, -1)[:-1])[0]
        falling_edge_idx = np.nonzero(high_samples[:-1] & np.roll(low_samples, -1)[:-1])[0]

        if len(rising_edge_idx) > 0:
            self.rising_edge = rising_edge_idx[0]

        if len(falling_edge_idx) > 0:
            self.falling_edge = falling_edge_idx[0]   

        # If on FIRST rising edge - record edge and increment samples counter then exit early
        if len(rising_edge_idx) == 1 and self.stateful_rising_edge == 0:
            print("*** exit early *** ")
            self.stateful_rising_edge = rising_edge_idx[0]
            self.stateful_index += samples.size
            return

        # Detects if a beep was sliced by end of chunk
        # To do - add logic to pass to next iteration number of samples between rising edge and chunk end
        # Then use that to make the calculations for beep duration and SNR 
        if len(rising_edge_idx) == 1 and len(falling_edge_idx) == 0:
            self.beep_slice = True
            self.distance_to_sample_end = len(samples)-rising_edge_idx[0]
            self.stateful_index += samples.size
            return
        
        # only run these lines if not on a self.beep_slice
        if not (self.beep_slice):
            if (len(rising_edge_idx) == 0 or len(falling_edge_idx) == 0):
                #print("both zero")
                self.stateful_index += samples.size
                return
            
            if (rising_edge_idx[0] > falling_edge_idx[0]) :
                falling_edge_idx = falling_edge_idx[1:]

            # Remove stray rising edge at the end
            if rising_edge_idx[-1] > falling_edge_idx[-1]:
                rising_edge_idx = rising_edge_idx[:-1]

        if (self.beep_slice):
            print("beep slice is true - calculating BPM")
            samples_between = (self.stateful_index-self.distance_to_sample_end) - self.stateful_rising_edge
            time_between = 1/sample_rate * (samples_between)
            BPM = 60 / time_between
            #print(f" BPM inside first if : {BPM}")
            self.stateful_rising_edge = self.stateful_index-self.distance_to_sample_end
        else:
            #print(f"beep slice is false - calculating BPM")
            samples_between =  (rising_edge_idx[0]+self.stateful_index) - self.stateful_rising_edge
            #print(f" new rising edge : {rising_edge_idx[0]+self.stateful_index}")
            #print(f" old rising edge :  {self.stateful_rising_edge}")
            time_between = 1/sample_rate * (samples_between)
            BPM = 60 / time_between
            self.stateful_rising_edge = self.stateful_index + rising_edge_idx[0]

        #print(f"printing prior to snr : {rising_edge_idx}")
        if (self.beep_slice):
            SNR = snr(samples_for_snr, self.rising_edge-5, self.falling_edge+5, self.beep_slice)
        else:
            SNR = snr(samples_for_snr, rising_edge_idx[0]-5, falling_edge_idx[0]+5, self.beep_slice)
        
        if (self.beep_slice):
            BEEP_DURATION = (falling_edge_idx[0]+self.distance_to_sample_end) / sample_rate
            self.beep_slice = False
        else:
            BEEP_DURATION = (falling_edge_idx[0]-rising_edge_idx[0]) / sample_rate
        
        print(f"  DATE : {datetime.now()} | BPM : {BPM: 5.2f} |  SNR : {SNR: 5.2f}  | BEEP_DURATION : {BEEP_DURATION: 5.4f} sec")
        self.bsm.process_input(BPM)

        # increment sample count
        self.stateful_index += samples.size
        self.rising_edge = 0
        self.falling_edge = 0

