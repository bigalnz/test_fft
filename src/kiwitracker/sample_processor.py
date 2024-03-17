from __future__ import annotations

import asyncio
import itertools
import logging
import math
import os
import platform
import statistics
import time
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy import signal

from kiwitracker.beep_state_machine import BeepStateMachine
from kiwitracker.chicktimerstatusdecoder import ChickTimerStatusDecoder
from kiwitracker.common import FloatArray, ProcessConfig, SamplesT
from kiwitracker.fasttelemtrydecoder import FastTelemetryDecoder

if platform.system() == "Linux":
    import gpsd


def snr(high_samples, low_samples):
    noise_pwr = np.var(low_samples)
    signal_pwr = np.var(high_samples)
    snr_db = 20 * np.log10(signal_pwr / noise_pwr)
    return snr_db


def clipping(high_samples):
    clipping_max = np.max(high_samples)
    return clipping_max


def dBFS(high_samples):
    pwr = np.square(high_samples)
    pwr = np.average(pwr)
    pwr_dbfs = pwr_dBFS = 10 * np.log10(pwr / 1)
    return pwr_dbfs


class SampleProcessor:
    config: ProcessConfig
    threshold: float = 0.2
    beep_duration: float = 0.017  # seconds
    stateful_index: int
    beep_slice: bool
    rising_edge: int
    falling_edge: int
    ct_state: bool
    snrlist: list
    dbfslist: list

    def __init__(self, config: ProcessConfig) -> None:
        self.config = config
        self.stateful_index = 0
        self._time_array = None
        self._fir = None
        self._phasor = None
        self.stateful_rising_edge = 0
        self.beep_slice = False
        self.first_half_of_sliced_beep = 0
        self.rising_edge = 0
        self.falling_edge = 0
        self.bsm = BeepStateMachine(config)
        # self.f = open('testing_file.fc32', 'wb')
        # self.f2 = open('ct_80.fc32', 'wb')

        if config.running_mode == "normal":
            gpsd.connect()

        self.sample_checker = 0
        # create logger
        self.logger = logging.getLogger("KiwiTracker")
        self.beep_idx = 0

        self.test_samp = np.array(0)
        self.save_flag = False
        self.i = 0
        SNR = 0.0
        DBFS = 0.0
        self.valid_intervals = [250, 750, 1250, 1750, 2000, 3000, 3750]
        self.valid_BPMs = [60 / (interval / 1000) for interval in self.valid_intervals]
        self.decoder = ChickTimerStatusDecoder()
        self.fast_telemetry_decoder = FastTelemetryDecoder()

        self.ct_state = False
        self.snrlist = []
        self.dbfslist = []

    @property
    def platform_property(self):
        return platform.system()

    @property
    def channel(self):
        """Channel Number from Freq"""
        return math.floor((self.config.carrier_freq - 160.11e6) / 0.01e6)

    @property
    def sample_rate(self):
        return self.config.sample_config.sample_rate

    @property
    def num_samples_to_process(self):
        return self.config.num_samples_to_process

    @property
    def carrier_freq(self):
        """Center frequency of the carrier wave to process (in Hz)"""
        return self.config.carrier_freq

    @carrier_freq.setter
    def carrier_freq(self, carrier_freq: float) -> None:
        self.config.carrier_freq = carrier_freq

    @property
    def freq_offset(self) -> float:
        """The offset (difference) between the sdr's center frequency and
        the :attr:`carrier_freq`
        """
        fc = self.config.sample_config.center_freq
        return fc - self.carrier_freq

    @freq_offset.setter
    def freq_offset(self, freq_offset: float) -> None:
        self.carrier_freq = self.config.sample_config.center_freq + freq_offset

    @property
    def time_array(self) -> FloatArray:
        t = self._time_array
        if t is None:
            t = np.arange(self.num_samples_to_process) / self.sample_rate
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
            p = np.exp(2j * np.pi * t * self.freq_offset)
            self._phasor = p
        return p

    @property
    def fft_size(self) -> int:
        # this makes sure there's at least 1 full chunk within each beep
        return int(self.beep_duration * self.sample_rate / 2)

    def find_beep_freq(self, samples):
        # print("find beep freq ran")
        # look for the presence of a beep within the chunk and :
        # (1) if beep found calculate the offset
        # (2) if beep not found iterate the counters and move on

        fft_size = self.fft_size
        f = np.linspace(self.sample_rate / -2, self.sample_rate / 2, fft_size)
        size = fft_size
        step = int(size // 1.1)
        samples_to_send_to_fft = [samples[i : i + size] for i in range(0, len(samples), step)]
        num_ffts = len(samples_to_send_to_fft)  # // is an integer division which rounds down
        beep_freqs = []

        i = 0
        for i in range(num_ffts):
            # fft = np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size]))) / fft_size
            fft = np.abs(np.fft.fftshift(np.fft.fft(samples_to_send_to_fft[i]))) / fft_size

            # plt.plot(fft)
            # plt.show()
            # print(f"{np.max(fft)/np.median(fft)}")

            if (np.max(fft) / np.median(fft)) > 20:
                # if np.max(fft) > fft_thresh:
                fft_freqs = np.linspace(self.sample_rate / -2, self.sample_rate / 2, fft_size)
                # plt.plot(fft_freqs, fft)
                # plt.show()
                # print(f"{np.median(fft)}")
                # print(f"{np.max(fft)}")
                beep_freqs.append(np.linspace(self.sample_rate / -2, self.sample_rate / 2, fft_size)[np.argmax(fft)])
                # beep_freqs.append(self.sample_rate/-2+np.argmax(fft)/fft_size*self.sample_rate) more efficent??

        if len(beep_freqs) != 0:
            bp = np.array(beep_freqs)
            bp = bp + self.config.sample_config.center_freq
            bp = bp.astype(np.int64)
            beep_freqs_singular = [
                statistics.mean(x) for _, x in itertools.groupby(sorted(bp), key=lambda f: (f + 5000) // 10000)
            ]
            self.logger.info(f"detected beep_freqs offsets is {beep_freqs}")
            self.logger.info(f"detected beep_freqs_singular offsets is {beep_freqs_singular}")

            # print(f"about to set freq_offset. beep_freqs[0] is {beep_freqs[0]} and the np.max(fft) is {np.max(fft)}")
            self.freq_offset = beep_freqs[0]
            self.config.carrier_freq = beep_freqs[0] + self.config.sample_config.center_freq
            return beep_freqs[0]

        return 0

    def process(self, samples: SamplesT):

        if self.freq_offset == 0:
            self.find_beep_freq(samples)

        if self.freq_offset == 0:
            return

        # record this file in the field - for testing log with IQ values
        # self.f.write(samples.astype(np.complex128))

        # **************************************************
        # Use this blockto save to file without pickle issue
        # **************************************************
        """if (self.stateful_index < 150000 and not self.save_flag):
            self.test_samp = np.append(self.test_samp, samples)
            #print(len(self.test_samp))
        elif(self.stateful_index > 150000 and not self.save_flag):
            #self.f2.write(self.test_samp.astype(np.complex128).tobytes, allow_pickle=False)
            #print(len(self.test_samp))
            np.save(self.f2, self.test_samp, allow_pickle=False)
            print("********* file saved ****************")
            self.save_flag = True"""

        t = self.time_array
        samples = samples * self.phasor
        # next two lines are band pass filter?
        h = self.fir
        samples = signal.convolve(samples, h, "same")
        # decimation
        # recalculation of sample rate due to decimation
        sample_rate = self.sample_rate / 100

        # record this file in the field - for testing log with IQ values
        # self.f.write(samples.astype(np.complex128))
        samples = samples[::100]
        samples = np.abs(samples)
        # smoothing
        samples = signal.convolve(samples, [1] * 10, "same") / 189

        # for testing - log to file
        # self.f.write(samples.astype(np.float32).tobytes())

        # Get a boolean array for all samples higher or lower than the threshold
        self.threshold = np.median(samples) * 3  # go just above noise floor
        low_samples = samples < self.threshold
        high_samples = samples >= self.threshold

        # Compute the rising edge and falling edges by comparing the current value to the next with
        # the boolean operator & (if both are true the result is true) and converting this to an index
        # into the current array
        rising_edge_idx = np.nonzero(low_samples[:-1] & np.roll(high_samples, -1)[:-1])[0]
        falling_edge_idx = np.nonzero(high_samples[:-1] & np.roll(low_samples, -1)[:-1])[0]

        if len(rising_edge_idx) > 0:
            # print(f"len rising edges idx {len(rising_edge_idx)}")
            self.rising_edge = rising_edge_idx[0]

        if len(falling_edge_idx) > 0:
            self.falling_edge = falling_edge_idx[0]

        # If no rising or falling edges detected - exit early increment counter and move on
        if len(rising_edge_idx) == 0 and len(falling_edge_idx) == 0:
            self.stateful_index += samples.size
            return

        # If on FIRST rising edge - record edge and increment samples counter then exit early
        if len(rising_edge_idx) == 1 and self.stateful_rising_edge == 0:
            # print("*** exit early *** ")
            self.stateful_rising_edge = rising_edge_idx[0]
            self.stateful_index += samples.size  # + 9
            return

        # Detects if a beep was sliced by end of chunk
        # Grabs the samples from rising edge of end of samples and passes them to next iteration using samples_between
        if len(rising_edge_idx) == 1 and len(falling_edge_idx) == 0:
            self.beep_idx = (
                rising_edge_idx[0] + self.stateful_index
            )  # gives you the index of the beep using running total
            self.beep_slice = True
            self.distance_to_sample_end = (
                len(samples) - rising_edge_idx[0]
            )  # gives you the distance from the end of samples chunk
            self.stateful_index += samples.size  # + 14
            self.first_half_of_sliced_beep = samples[rising_edge_idx[0] :]
            # print("beep slice condition ln 229 is true - calculating BPM")
            return

        # only run these lines if not on a self.beep_slice
        # not on a beep slice, but missing either a rising or falling edge
        # is this ever true?
        if not (self.beep_slice):
            if len(rising_edge_idx) == 0 or len(falling_edge_idx) == 0:
                self.stateful_index += samples.size  # + 9
                return

            if rising_edge_idx[0] > falling_edge_idx[0]:
                falling_edge_idx = falling_edge_idx[1:]

            # Remove stray rising edge at the end
            if rising_edge_idx[-1] > falling_edge_idx[-1]:
                rising_edge_idx = rising_edge_idx[:-1]

        # BPM CALCUATIONS
        if self.beep_slice:
            samples_between = (self.stateful_index - self.distance_to_sample_end) - self.stateful_rising_edge
            time_between = 1 / sample_rate * (samples_between)
            BPM = 60 / time_between
            self.stateful_rising_edge = self.stateful_index - self.distance_to_sample_end
        else:
            samples_between = (rising_edge_idx[0] + self.stateful_index) - self.stateful_rising_edge
            self.beep_idx = rising_edge_idx[0] + self.stateful_index
            time_between = 1 / sample_rate * (samples_between)
            BPM = 60 / time_between
            self.stateful_rising_edge = self.stateful_index + rising_edge_idx[0]

        # GET HIGH AND LOW SAMPLES IN THERE OWN ARRAYS FOR CALC ON SNR, DBFS, CLIPPING
        if self.beep_slice:
            high_samples = np.concatenate((self.first_half_of_sliced_beep, samples[: self.falling_edge]))
            low_samples = samples[self.falling_edge :]
        else:
            high_samples = samples[self.rising_edge : self.falling_edge]
            low_samples = np.concatenate((samples[: self.rising_edge], samples[self.falling_edge :]))

        SNR = snr(high_samples, low_samples)
        DBFS = dBFS(high_samples)
        CLIPPING = clipping(high_samples)

        # BEEP DURATION
        if self.beep_slice:
            if len(falling_edge_idx) != 0:
                BEEP_DURATION = (falling_edge_idx[0] + self.distance_to_sample_end) / sample_rate
                # self.beep_slice = False
            else:
                BEEP_DURATION = 0
                # print(f"setting beep slice false on ln 273")
                # self.beep_slice = False
        else:
            BEEP_DURATION = (falling_edge_idx[0] - rising_edge_idx[0]) / sample_rate

        # GET GPS INFO FROM GPSD
        if self.config.running_mode == "normal":
            packet = gpsd.get_current()
            latitude = packet.lat
            longitude = packet.lon
        else:
            # use a default value for now
            latitude = -36.8807
            longitude = 174.924

        # print(f"  DATE : {datetime.now()} | BPM : {BPM: 5.2f} |  SNR : {SNR: 5.2f}  | BEEP_DURATION : {BEEP_DURATION: 5.4f} sec | POS : {latitude} {longitude}")
        self.logger.info(
            f" BPM : {BPM: 5.2f} | PWR : {DBFS or 0:5.2f} dBFS | MAG : {CLIPPING: 5.3f} | BEEP_DURATION : {BEEP_DURATION: 5.4f}s | SNR : {SNR: 5.2f} | POS : {latitude} {longitude}"
        )

        self.fast_telemetry_decoder.send(BPM)

        # Send normalized BPMs to ChickTImerStatusDecoder
        ChickTimerStatus = self.decoder.current_state
        normalized_BPMs = min(self.valid_BPMs, key=lambda x: abs(x - BPM))
        self.decoder.send(normalized_BPMs)

        # Check for CT start by looking at change of state
        if ChickTimerStatus is self.decoder.background and self.decoder.current_state is self.decoder.tens_digit:
            print(" **** CT START *****")
            self.decoder.ct.start_date_time = datetime.now()
            self.ct_state = True
        # self.logger.info(f" Normalised BP : {normalized_BPMs} Input BPM : {BPM: 5.2f} current decoder state :{self.decoder.current_state}" )

        if self.ct_state:
            self.snrlist.append(SNR)
            self.dbfslist.append(DBFS)

        if normalized_BPMs == 20:
            print(self.decoder.ct)

        # Check for CT end by looking at change of values
        if ChickTimerStatus is self.decoder.ones_digit and self.decoder.current_state is self.decoder.background:
            print(" **** CT FINISH *****")
            self.decoder.ct.lat = latitude
            self.decoder.ct.lon = longitude
            self.decoder.ct.finish_date_time = datetime.now()
            self.decoder.ct.carrier_freq = self.carrier_freq
            self.decoder.ct.channel = self.channel
            print(self.snrlist)
            print(min(self.snrlist))
            self.decoder.ct.snr.min = round(min(self.snrlist), 2)
            self.decoder.ct.snr.max = round(max(self.snrlist), 2)
            self.decoder.ct.snr.mean = round(statistics.mean(self.snrlist), 2)
            self.decoder.ct.dbfs.min = round(min(self.dbfslist), 2)
            self.decoder.ct.dbfs.max = round(max(self.dbfslist), 2)
            self.decoder.ct.dbfs.mean = round(statistics.mean(self.dbfslist), 2)
            print(f" This is the new ct : {self.decoder.ct.toJSON()}")
            # send to db or whatever here
            # reset everything
            self.decoder.ct.__init__()
            self.decoder.ct.status.__init__(0, 0, 0, 0, 0, 0, 0, 0)
            self.dbfslist.clear()
            self.snrlist.clear()
            self.ct_state = False

        # Send to Old Finite State Machine
        # self.bsm.process_input(BPM, SNR, DBFS, latitude, longitude)

        # increment sample count
        self.beep_slice = False
        self.stateful_index += samples.size  # + 14
        self.rising_edge = 0
        self.falling_edge = 0

        # reset these?
        # high_samples
        # low_samples
