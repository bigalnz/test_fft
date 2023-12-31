from __future__ import annotations
from typing import TYPE_CHECKING
import asyncio
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt
from scipy import signal
import time

from kiwitracker.common import SamplesT, FloatArray, ProcessConfig


def snr(samples, rising_edge_idx, falling_edge_idx):
    noise_pwr = np.var( np.concatenate([samples[:rising_edge_idx], samples[falling_edge_idx:]]) )
    signal_pwr = np.var ( samples[rising_edge_idx:falling_edge_idx] )
    snr_db = 10 * np.log10 ( signal_pwr / noise_pwr )
    return snr_db

class SampleProcessor:
    config: ProcessConfig
    threshold: float = 0.6
    beep_duration: float = 0.017 # seconds
    stateful_index: int

    def __init__(self, config: ProcessConfig) -> None:
        self.config = config
        self.stateful_index = 0
        self._time_array = None
        self._fir = None
        self._phasor = None
        self.stateful_rising_edge = 0

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

        start_time = time.time()
        fft_size = self.fft_size
        f = np.linspace(self.sample_rate/-2, self.sample_rate/2, fft_size)
        num_ffts = len(samples) // fft_size # // is an integer division which rounds down
        fft_thresh = 0.1
        beep_freqs = []
        for i in range(num_ffts):
            fft = np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size]))) / fft_size
            if np.max(fft) > fft_thresh:
                beep_freqs.append(np.linspace(self.sample_rate/-2, self.sample_rate/2, fft_size)[np.argmax(fft)])
        finish_time = time.time()

        # if not beeps increment and exit early
        if len(beep_freqs) == 0:
            self.stateful_index += (samples.size/100) + 14
            return
        
        #plt.plot(f,fft)
        # print(beep_freqs)
        # #plt.show()

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
        # max_samp = np.max(samples)

        # samples /= np.max(samples)
        #plt.plot(samples)
        #plt.show()

        # Get a boolean array for all samples higher or lower than the threshold
        low_samples = samples < self.threshold
        high_samples = samples >= self.threshold

        # Compute the rising edge and falling edges by comparing the current value to the next with
        # the boolean operator & (if both are true the result is true) and converting this to an index
        # into the current array
        rising_edge_idx = np.nonzero(low_samples[:-1] & np.roll(high_samples, -1)[:-1])[0]
        falling_edge_idx = np.nonzero(high_samples[:-1] & np.roll(low_samples, -1)[:-1])[0]

        if len(rising_edge_idx) == 0 or len(falling_edge_idx) == 0:
            self.stateful_index += samples.size + 14
            return
        #print(f"passed len test for idx's")
        if rising_edge_idx[0] > falling_edge_idx[0]:
            falling_edge_idx = falling_edge_idx[1:]

        # Remove stray rising edge at the end
        if rising_edge_idx[-1] > falling_edge_idx[-1]:
            rising_edge_idx = rising_edge_idx[:-1]

        # rising_edge_diff = np.diff(rising_edge_idx)
        # time_between_rising_edge = sample_rate / rising_edge_diff * 60

        # pulse_widths = falling_edge_idx - rising_edge_idx
        # rssi_idxs = list(np.arange(r, r + p) for r, p in zip(rising_edge_idx, pulse_widths))
        # rssi = [np.mean(samples[r]) * max_samp for r in rssi_idxs]

        # for t, r in zip(time_between_rising_edge, rssi):
        #     print(f"BPM: {t:.02f}")
        #     print(f"rssi: {r:.02f}")
        # self.stateful_index += len(samples)
        # print(f"stateful index : {self.stateful_index}")

        #print(f"stateful rising edge : {self.stateful_rising_edge}")
        #print(f" samples size : {samples.size}")
        #print(f"rising edge idx [0] : {rising_edge_idx[0]}")
        #print(f" stateful index : {self.stateful_index}")
        #print("*****************************************")

        samples_between =  (rising_edge_idx[0]+self.stateful_index) - self.stateful_rising_edge
        time_between = 1/sample_rate * samples_between
        BPM = 60 / time_between
        self.stateful_rising_edge = self.stateful_index + rising_edge_idx[0]
        SNR = snr(samples_for_snr, rising_edge_idx[0]-5, falling_edge_idx[0]+5)
        BEEP_DURATION = (falling_edge_idx[0]-rising_edge_idx[0]) / sample_rate
        
        print(f" BPM : {BPM: 5.2f} |  SNR : {SNR: 5.2f}  | BEEP_DURATION : {BEEP_DURATION: 5.4f} sec")
        # increment sample count
        self.stateful_index += samples.size + 14
