from __future__ import annotations

import asyncio
import itertools
import logging
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy import signal

from kiwitracker.chicktimerstatusdecoder import ChickTimerStatusDecoder
from kiwitracker.common import (FloatArray, ProcessConfig, ProcessResult,
                                SamplesT)
from kiwitracker.exceptions import CarrierFrequencyNotFound
from kiwitracker.fasttelemetrydecoder import FastTelemetryDecoder

if platform.system() == "Linux":
    import gpsd

from functools import lru_cache


@lru_cache(maxsize=1)
def fir() -> np.ndarray:
    return signal.firwin(501, 0.02, pass_zero=True)


@lru_cache(maxsize=5)
def time_array(num_samples_to_process, sample_rate) -> np.ndarray:
    return np.arange(num_samples_to_process) / sample_rate


@lru_cache(maxsize=5)
def phasor(num_samples_to_process, sample_rate, freq_offset) -> np.ndarray:
    return np.exp(2j * np.pi * time_array(num_samples_to_process, sample_rate) * freq_offset)


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
    pwr_dbfs = 10 * np.log10(pwr / 1)
    return pwr_dbfs


class SampleProcessor:
    config: ProcessConfig
    ct_state: bool
    snrlist: list
    dbfslist: list

    def __init__(self, config: ProcessConfig) -> None:
        self.config = config

        if config.running_mode == "normal":
            gpsd.connect()

        self.sample_checker = 0
        # create logger
        self.logger = logging.getLogger("KiwiTracker")
        # self.beep_idx = 0

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

    @staticmethod
    async def find_beep_freq_2(samples_queue, logger, pc: ProcessConfig, N: int):
        out = []

        for i in range(N):

            logger.info(f"Scanning chunk no. {i+1}...")

            # print("find beep freq ran")
            # look for the presence of a beep within the chunk and :
            # (1) if beep found calculate the offset
            # (2) if beep not found iterate the counters and move on

            # TO DO
            # This function only to be called when --scan is given on command line
            # This function needs to operated on the first 12 chunks (3 seconds) of data
            # The detected beep array should reject any value not between 160110000 and 161120000

            samples = await samples_queue.get()

            # f = np.linspace(sample_rate / -2, sample_rate / 2, fft_size)
            size = pc.fft_size
            step = int(size // 1.1)
            samples_to_send_to_fft = [samples[i : i + size] for i in range(0, len(samples), step)]
            num_ffts = len(samples_to_send_to_fft)  # // is an integer division which rounds down
            beep_freqs = []

            i = 0
            for i in range(num_ffts):
                # fft = np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size]))) / fft_size
                fft = np.abs(np.fft.fftshift(np.fft.fft(samples_to_send_to_fft[i]))) / pc.fft_size

                # plt.plot(fft)
                # plt.show()
                # print(f"{np.max(fft)/np.median(fft)}")

                # DC Spike removal
                fft[len(fft) // 2] = np.mean(fft[(len(fft) // 2) - 10 : (len(fft) // 2) - 3])

                if (np.max(fft) / np.median(fft)) > 20:
                    # if np.max(fft) > fft_thresh:
                    fft_freqs = np.linspace(pc.sample_rate / -2, pc.sample_rate / 2, pc.fft_size)
                    fft_freqs = fft_freqs + pc.sample_config.center_freq
                    # plt.plot(fft_freqs, fft)
                    # plt.show()
                    # print(f"{np.median(fft)}")
                    # print(f"{np.max(fft)}")

                    # This needs to be changes to np.argwhere((np.max(fft) / np.median(fft)) > 20)
                    # So it adds all freqs detected over the threshold
                    #noise_floor = np.median(fft) * 5
                    #beep_freqs.append(np.linspace(pc.sample_rate / -2, pc.sample_rate / 2, pc.fft_size)[np.argmax(fft)])
                    beep_freqs.extend(np.linspace(pc.sample_rate / -2, pc.sample_rate / 2, pc.fft_size)[signal.find_peaks(fft, prominence=0.3)[0]])
                    # beep_freqs.append(self.sample_rate/-2+np.argmax(fft)/fft_size*self.sample_rate) more efficent??

            if len(beep_freqs) != 0:
                logger.info(f"detected beep_freqs offsets is {beep_freqs}")
                logger.info(f"appending to output {beep_freqs[0] + pc.sample_config.center_freq}")
                beep_freqs = np.array(beep_freqs) + pc.sample_config.center_freq
                beep_freqs[(beep_freqs >= 160100110 ) & (beep_freqs <= 161120000)]
                beep_freqs = beep_freqs.tolist()
                out.extend(beep_freqs)

            samples_queue.task_done()

        out = [statistics.mean(x) for _, x in itertools.groupby(sorted(out), key=lambda f: (f + 5000) // 10000)]
        return [int(out) for out in out ]

    @staticmethod
    def decimate_samples(samples, previous_samples, pc: ProcessConfig, chunk_count):
        #########################################################################
        samples = samples * phasor(pc.num_samples_to_process, pc.sample_rate, pc.freq_offset)[: samples.size]
        # next two lines are band pass filter?
        samples = signal.convolve(samples, fir(), "same")
        # decimation
        samples = np.abs(samples[::100])

        unsmoothed_samples = samples
        # smoothing - smoothed samples are good for beep detection only.
        # samples = signal.convolve(samples, [1] * 189, "valid") / 189
        samples = signal.convolve(np.concatenate((previous_samples[-188:], samples)), [1] * 189, "valid") / 189
        return samples, pc.sample_rate / 100, unsmoothed_samples

    @staticmethod
    def get_rising_falling_indices(samples, unsmoothed_samples):
        # Get a boolean array for all samples higher or lower than the threshold
        threshold = np.median(samples) * 3  # go just above noise floor - use smoothed samples for beep detection

        # can exit this chunk here early if nothing detected above threshold
        # if not True in samples >= threshold:
        # iterate counters
        # return (go to next chunk)

        # if got to this point - then there is probably a signal in the chunk
        low_samples = samples < threshold
        high_samples = samples >= threshold

        # TODO: exit early here

        # Compute the rising edge and falling edges by comparing the current value to the next with
        # the boolean operator & (if both are true the result is true) and converting this to an index
        # into the current array
        rising_edge_idx = np.nonzero(low_samples[:-1] & np.roll(high_samples, -1)[:-1])[0]
        falling_edge_idx = np.nonzero(high_samples[:-1] & np.roll(low_samples, -1)[:-1])[0]

        return rising_edge_idx, falling_edge_idx

    # @staticmethod
    # def detect_beep_coro(pc):
    #     samples, sample_rate = yield

    #     while True:
    #         rising_edge_idx, falling_edge_idx = SampleProcessor.get_rising_falling_indices(samples)

    #         # not rising, nor falling, return None and wait for other sample
    #         if not rising_edge_idx and not falling_edge_idx:
    #             samples, sample_rate = yield None
    #         # are we in sliced beep?
    #         elif len(rising_edge_idx) == 1 and len(falling_edge_idx) == 0:
    #             rising_idx = rising_edge_idx[0]
    #             first_half_of_sliced_beep = samples[rising_edge_idx[0] :]
    #             # yes, so return None and wait for falling
    #             while True:
    #                 samples, sample_rate = yield None

    async def process_sample(self, pc: ProcessConfig, samples_queue: asyncio.Queue, out_queue: asyncio.Queue) -> None:
        rising_edge = 0
        falling_edge = 0
        stateful_index = 0
        stateful_rising_edge = 0
        beep_slice = False
        # beep_idx = 0
        chunk_count = 0
        tot_samp_count = 0
        previous_samples = [0] * 188

        distance_to_sample_end = None
        first_half_of_sliced_beep = 0

        if pc.carrier_freq is None:
            self.logger.info("Carrier frequency not set - start scanning...")

            frequencies = await self.find_beep_freq_2(samples_queue, self.logger, pc, N=13)
            if not frequencies:
                self.logger.error("Not single frequency detected, exiting...")
                raise CarrierFrequencyNotFound()
            else:
                self.logger.info(f"Frequencies detected: {frequencies}")
                self.logger.info(f"Picking first one: {frequencies[0]}")
                pc.carrier_freq = frequencies[0]

        while True:
            samples = await samples_queue.get()

            # do we need to quit? E.g. we processed all chunks from test file
            if samples is None:
                samples_queue.task_done()
                break

            chunk_count += 1
            tot_samp_count = tot_samp_count + len(samples)

            # print(f"Received sample: {samples.size}")
            samples, sample_rate, unsmoothed_samples = self.decimate_samples(samples, previous_samples, pc, chunk_count)
            previous_samples = unsmoothed_samples

            # ... = dbc.send(samples, sample_rate)

            rising_edge_idx, falling_edge_idx = self.get_rising_falling_indices(samples, unsmoothed_samples)

            # if len(rising_edge_idx) > 0 or len(falling_edge_idx) > 0:
            print(rising_edge_idx, falling_edge_idx, beep_slice)
            print(f"Chunk Count : {chunk_count}")

            if len(rising_edge_idx) > 0:
                # print(f"len rising edges idx {len(rising_edge_idx)}")
                rising_edge = rising_edge_idx[0]

            if len(falling_edge_idx) > 0:
                falling_edge = falling_edge_idx[0]

            # If no rising or falling edges detected - exit early increment counter and move on
            if len(rising_edge_idx) == 0 and len(falling_edge_idx) == 0:
                stateful_index += samples.size

                samples_queue.task_done()
                continue

            # If on FIRST rising edge - record edge and increment samples counter then exit early
            # if len(rising_edge_idx) == 1 and stateful_rising_edge == 0:
            #     # print("*** exit early *** ")
            #     stateful_rising_edge = rising_edge_idx[0]
            #     stateful_index += samples.size  # + 9

            #     samples_queue.task_done()
            #     continue

            # Detects if a beep was sliced by end of chunk
            # Grabs the samples from rising edge of end of samples and passes them to next iteration using samples_between
            if len(rising_edge_idx) == 1 and len(falling_edge_idx) == 0:
                # beep_idx = rising_edge_idx[0] + stateful_index  # gives you the index of the beep using running total
                beep_slice = True
                distance_to_sample_end = (
                    len(samples) - rising_edge_idx[0]
                )  # gives you the distance from the end of samples chunk
                stateful_index += samples.size  # + 14
                first_half_of_sliced_beep = samples[rising_edge_idx[0] :]
                # print("beep slice condition ln 229 is true - calculating BPM")

                samples_queue.task_done()
                continue

            # only run these lines if not on a self.beep_slice
            # not on a beep slice, but missing either a rising or falling edge
            # is this ever true?
            if not (beep_slice):
                if len(rising_edge_idx) == 0 or len(falling_edge_idx) == 0:
                    stateful_index += samples.size  # + 9

                    samples_queue.task_done()
                    continue

                if rising_edge_idx[0] > falling_edge_idx[0]:
                    falling_edge_idx = falling_edge_idx[1:]

                # Remove stray rising edge at the end
                if rising_edge_idx[-1] > falling_edge_idx[-1]:
                    rising_edge_idx = rising_edge_idx[:-1]

            # BPM CALCUATIONS
            if beep_slice:
                samples_between = (stateful_index - distance_to_sample_end) - stateful_rising_edge
                time_between = 1 / sample_rate * (samples_between)
                BPM = 60 / time_between
                stateful_rising_edge = stateful_index - distance_to_sample_end
            else:
                samples_between = (rising_edge_idx[0] + stateful_index) - stateful_rising_edge
                # beep_idx = rising_edge_idx[0] + stateful_index
                time_between = 1 / sample_rate * (samples_between)
                BPM = 60 / time_between
                stateful_rising_edge = stateful_index + rising_edge_idx[0]

            # GET HIGH AND LOW SAMPLES IN THERE OWN ARRAYS FOR CALC ON SNR, DBFS, CLIPPING
            if beep_slice:
                high_samples = np.concatenate((first_half_of_sliced_beep, samples[:falling_edge]))
                low_samples = samples[falling_edge:]
            else:
                high_samples = samples[rising_edge:falling_edge]
                low_samples = np.concatenate((samples[:rising_edge], samples[falling_edge:]))

            SNR = snr(high_samples, low_samples)
            DBFS = dBFS(high_samples)
            CLIPPING = clipping(high_samples)

            # BEEP DURATION
            if beep_slice:
                if len(falling_edge_idx) != 0:
                    BEEP_DURATION = ((falling_edge_idx[0] + distance_to_sample_end) / sample_rate) / 1.8
                    # self.beep_slice = False
                else:
                    BEEP_DURATION = 0
                    # print(f"setting beep slice false on ln 273")
                    # self.beep_slice = False
            else:
                BEEP_DURATION = ((falling_edge_idx[0] - rising_edge_idx[0]) / sample_rate) / 1.8

            # GET GPS INFO FROM GPSD
            if self.config.running_mode == "normal":
                packet = gpsd.get_current()
                latitude = packet.lat
                longitude = packet.lon
            else:
                # use a default value for now
                latitude = -36.8807
                longitude = 174.924

            print(f"{rising_edge=} {falling_edge=}")

            # print(f"  DATE : {datetime.now()} | BPM : {BPM: 5.2f} |  SNR : {SNR: 5.2f}  | BEEP_DURATION : {BEEP_DURATION: 5.4f} sec | POS : {latitude} {longitude}")
            self.logger.info(
                f" BPM : {BPM: 5.2f} | PWR : {DBFS or 0:5.2f} dBFS | MAG : {CLIPPING: 5.3f} | BEEP_DURATION : {BEEP_DURATION: 5.4f}s | SNR : {SNR: 5.2f} | POS : {latitude} {longitude}"
            )

            #########################################################################

            res = ProcessResult(datetime.now(), BPM, DBFS, CLIPPING, BEEP_DURATION, SNR, latitude, longitude)
            print(res)

            await out_queue.put(res)

            beep_slice = False
            stateful_index += samples.size  # + 14
            rising_edge = 0
            falling_edge = 0

            samples_queue.task_done()

    async def process_2(self, pc: ProcessConfig, samples_queue: asyncio.Queue, out_queue: asyncio.Queue) -> None:

        # first, find frequency offset (if not set via command line arguments)
        # Add logic - If --scan is passed then send the first 3 seconds of samples to be scanned
        # if pc.freq_offset == 0:
        #     pc.carrier_freq = await SampleProcessor.find_beep_freq_2(
        #         samples_queue=samples_queue, logger=self.logger, pc=pc
        #     )

        await self.process_sample(pc, samples_queue, out_queue)

        # while True:
        #     data = await samples_queue.get()
        #     print(f"Received sample: {data.size}")

        #     samples_queue.task_done()

    # def process(self, samples: SamplesT):

    #     if self.freq_offset == 0:
    #         self.find_beep_freq(samples)

    #     if self.freq_offset == 0:
    #         return

    #     # record this file in the field - for testing log with IQ values
    #     # self.f.write(samples.astype(np.complex128))

    #     # **************************************************
    #     # Use this blockto save to file without pickle issue
    #     # **************************************************
    #     """if (self.stateful_index < 150000 and not self.save_flag):
    #         self.test_samp = np.append(self.test_samp, samples)
    #         #print(len(self.test_samp))
    #     elif(self.stateful_index > 150000 and not self.save_flag):
    #         #self.f2.write(self.test_samp.astype(np.complex128).tobytes, allow_pickle=False)
    #         #print(len(self.test_samp))
    #         np.save(self.f2, self.test_samp, allow_pickle=False)
    #         print("********* file saved ****************")
    #         self.save_flag = True"""

    #     t = self.time_array
    #     samples = samples * self.phasor
    #     # next two lines are band pass filter?
    #     h = self.fir
    #     samples = signal.convolve(samples, h, "same")
    #     # decimation
    #     # recalculation of sample rate due to decimation
    #     sample_rate = self.sample_rate / 100

    #     # record this file in the field - for testing log with IQ values
    #     # self.f.write(samples.astype(np.complex128))
    #     samples = samples[::100]
    #     samples = np.abs(samples)
    #     # smoothing
    #     samples = signal.convolve(samples, [1] * 10, "same") / 189

    #     # for testing - log to file
    #     # self.f.write(samples.astype(np.float32).tobytes())

    #     # Get a boolean array for all samples higher or lower than the threshold
    #     self.threshold = np.median(samples) * 3  # go just above noise floor
    #     low_samples = samples < self.threshold
    #     high_samples = samples >= self.threshold

    #     # Compute the rising edge and falling edges by comparing the current value to the next with
    #     # the boolean operator & (if both are true the result is true) and converting this to an index
    #     # into the current array
    #     rising_edge_idx = np.nonzero(low_samples[:-1] & np.roll(high_samples, -1)[:-1])[0]
    #     falling_edge_idx = np.nonzero(high_samples[:-1] & np.roll(low_samples, -1)[:-1])[0]

    #     if len(rising_edge_idx) > 0:
    #         # print(f"len rising edges idx {len(rising_edge_idx)}")
    #         self.rising_edge = rising_edge_idx[0]

    #     if len(falling_edge_idx) > 0:
    #         self.falling_edge = falling_edge_idx[0]

    #     # If no rising or falling edges detected - exit early increment counter and move on
    #     if len(rising_edge_idx) == 0 and len(falling_edge_idx) == 0:
    #         self.stateful_index += samples.size
    #         return

    #     # If on FIRST rising edge - record edge and increment samples counter then exit early
    #     if len(rising_edge_idx) == 1 and self.stateful_rising_edge == 0:
    #         # print("*** exit early *** ")
    #         self.stateful_rising_edge = rising_edge_idx[0]
    #         self.stateful_index += samples.size  # + 9
    #         return

    #     # Detects if a beep was sliced by end of chunk
    #     # Grabs the samples from rising edge of end of samples and passes them to next iteration using samples_between
    #     if len(rising_edge_idx) == 1 and len(falling_edge_idx) == 0:
    #         self.beep_idx = (
    #             rising_edge_idx[0] + self.stateful_index
    #         )  # gives you the index of the beep using running total
    #         self.beep_slice = True
    #         self.distance_to_sample_end = (
    #             len(samples) - rising_edge_idx[0]
    #         )  # gives you the distance from the end of samples chunk
    #         self.stateful_index += samples.size  # + 14
    #         self.first_half_of_sliced_beep = samples[rising_edge_idx[0] :]
    #         # print("beep slice condition ln 229 is true - calculating BPM")
    #         return

    #     # only run these lines if not on a self.beep_slice
    #     # not on a beep slice, but missing either a rising or falling edge
    #     # is this ever true?
    #     if not (self.beep_slice):
    #         if len(rising_edge_idx) == 0 or len(falling_edge_idx) == 0:
    #             self.stateful_index += samples.size  # + 9
    #             return

    #         if rising_edge_idx[0] > falling_edge_idx[0]:
    #             falling_edge_idx = falling_edge_idx[1:]

    #         # Remove stray rising edge at the end
    #         if rising_edge_idx[-1] > falling_edge_idx[-1]:
    #             rising_edge_idx = rising_edge_idx[:-1]

    #     # BPM CALCUATIONS
    #     if self.beep_slice:
    #         samples_between = (self.stateful_index - self.distance_to_sample_end) - self.stateful_rising_edge
    #         time_between = 1 / sample_rate * (samples_between)
    #         BPM = 60 / time_between
    #         self.stateful_rising_edge = self.stateful_index - self.distance_to_sample_end
    #     else:
    #         samples_between = (rising_edge_idx[0] + self.stateful_index) - self.stateful_rising_edge
    #         self.beep_idx = rising_edge_idx[0] + self.stateful_index
    #         time_between = 1 / sample_rate * (samples_between)
    #         BPM = 60 / time_between
    #         self.stateful_rising_edge = self.stateful_index + rising_edge_idx[0]

    #     # GET HIGH AND LOW SAMPLES IN THERE OWN ARRAYS FOR CALC ON SNR, DBFS, CLIPPING
    #     if self.beep_slice:
    #         high_samples = np.concatenate((self.first_half_of_sliced_beep, samples[: self.falling_edge]))
    #         low_samples = samples[self.falling_edge :]
    #     else:
    #         high_samples = samples[self.rising_edge : self.falling_edge]
    #         low_samples = np.concatenate((samples[: self.rising_edge], samples[self.falling_edge :]))

    #     SNR = snr(high_samples, low_samples)
    #     DBFS = dBFS(high_samples)
    #     CLIPPING = clipping(high_samples)

    #     # BEEP DURATION
    #     if self.beep_slice:
    #         if len(falling_edge_idx) != 0:
    #             BEEP_DURATION = (falling_edge_idx[0] + self.distance_to_sample_end) / sample_rate
    #             # self.beep_slice = False
    #         else:
    #             BEEP_DURATION = 0
    #             # print(f"setting beep slice false on ln 273")
    #             # self.beep_slice = False
    #     else:
    #         BEEP_DURATION = (falling_edge_idx[0] - rising_edge_idx[0]) / sample_rate

    #     # GET GPS INFO FROM GPSD
    #     if self.config.running_mode == "normal":
    #         packet = gpsd.get_current()
    #         latitude = packet.lat
    #         longitude = packet.lon
    #     else:
    #         # use a default value for now
    #         latitude = -36.8807
    #         longitude = 174.924

    #     # print(f"  DATE : {datetime.now()} | BPM : {BPM: 5.2f} |  SNR : {SNR: 5.2f}  | BEEP_DURATION : {BEEP_DURATION: 5.4f} sec | POS : {latitude} {longitude}")
    #     self.logger.info(
    #         f" BPM : {BPM: 5.2f} | PWR : {DBFS or 0:5.2f} dBFS | MAG : {CLIPPING: 5.3f} | BEEP_DURATION : {BEEP_DURATION: 5.4f}s | SNR : {SNR: 5.2f} | POS : {latitude} {longitude}"
    #     )

    #     self.fast_telemetry_decoder.send(BPM)

    #     # Send normalized BPMs to ChickTImerStatusDecoder
    #     ChickTimerStatus = self.decoder.current_state
    #     normalized_BPMs = min(self.valid_BPMs, key=lambda x: abs(x - BPM))
    #     self.decoder.send(normalized_BPMs)

    #     # Check for CT start by looking at change of state
    #     if ChickTimerStatus is self.decoder.background and self.decoder.current_state is self.decoder.tens_digit:
    #         print(" **** CT START *****")
    #         self.decoder.ct.start_date_time = datetime.now()
    #         self.ct_state = True
    #     # self.logger.info(f" Normalised BP : {normalized_BPMs} Input BPM : {BPM: 5.2f} current decoder state :{self.decoder.current_state}" )

    #     if self.ct_state:
    #         self.snrlist.append(SNR)
    #         self.dbfslist.append(DBFS)

    #     if normalized_BPMs == 20:
    #         print(self.decoder.ct)

    #     # Check for CT end by looking at change of values
    #     if ChickTimerStatus is self.decoder.ones_digit and self.decoder.current_state is self.decoder.background:
    #         print(" **** CT FINISH *****")
    #         self.decoder.ct.lat = latitude
    #         self.decoder.ct.lon = longitude
    #         self.decoder.ct.finish_date_time = datetime.now()
    #         self.decoder.ct.carrier_freq = self.carrier_freq
    #         self.decoder.ct.channel = self.channel
    #         print(self.snrlist)
    #         print(min(self.snrlist))
    #         self.decoder.ct.snr.min = round(min(self.snrlist), 2)
    #         self.decoder.ct.snr.max = round(max(self.snrlist), 2)
    #         self.decoder.ct.snr.mean = round(statistics.mean(self.snrlist), 2)
    #         self.decoder.ct.dbfs.min = round(min(self.dbfslist), 2)
    #         self.decoder.ct.dbfs.max = round(max(self.dbfslist), 2)
    #         self.decoder.ct.dbfs.mean = round(statistics.mean(self.dbfslist), 2)
    #         print(f" This is the new ct : {self.decoder.ct.toJSON()}")
    #         # send to db or whatever here
    #         # reset everything
    #         self.decoder.ct.__init__()
    #         self.decoder.ct.status.__init__(0, 0, 0, 0, 0, 0, 0, 0)
    #         self.dbfslist.clear()
    #         self.snrlist.clear()
    #         self.ct_state = False

    #     # Send to Old Finite State Machine
    #     # self.bsm.process_input(BPM, SNR, DBFS, latitude, longitude)

    #     # increment sample count
    #     self.beep_slice = False
    #     self.stateful_index += samples.size  # + 14
    #     self.rising_edge = 0
    #     self.falling_edge = 0

    #     # reset these?
    #     # high_samples
    #     # low_samples
