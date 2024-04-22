from __future__ import annotations

import asyncio
import itertools
import logging
import math
import statistics
from datetime import datetime
from functools import lru_cache
from typing import AsyncIterator

import numpy as np
# TODO: This plotting stuff shouldn't propably be here:
from matplotlib import pyplot as plt
from scipy import signal

from kiwitracker.common import CTResult, ProcessConfig, ProcessResult
from kiwitracker.exceptions import ChickTimerProcessingError

logger = logging.getLogger("KiwiTracker")


@lru_cache(maxsize=1)
def fir() -> np.ndarray:
    return signal.firwin(501, 0.02, pass_zero=True)


@lru_cache(maxsize=5)
def time_array(num_samples_to_process: int, sample_rate: int) -> np.ndarray:
    return np.arange(num_samples_to_process) / sample_rate


@lru_cache(maxsize=5)
def phasor(num_samples_to_process: int, sample_rate: int, freq_offset: int) -> np.ndarray:
    return np.exp(2j * np.pi * time_array(num_samples_to_process, sample_rate) * freq_offset)


@lru_cache(maxsize=5)
def fft_freqs_array(sample_rate: int, fft_size: int) -> np.ndarray:
    return np.linspace(sample_rate / -2, sample_rate / 2, fft_size)


def channel(carrier_freq: float) -> int:
    """Channel Number from Freq"""
    return math.floor((carrier_freq - 160.11e6) / 0.01e6)


def snr(high_samples: np.ndarray, low_samples: np.ndarray) -> float:
    noise_pwr = np.var(low_samples)
    signal_pwr = np.var(high_samples)
    snr_db = 20 * np.log10(signal_pwr / noise_pwr)
    return snr_db


def clipping(high_samples: np.ndarray) -> float:
    clipping_max = np.max(high_samples)
    return clipping_max


def dBFS(high_samples: np.ndarray) -> float:
    pwr = np.square(high_samples)
    pwr = np.average(pwr)
    pwr_dbfs = 10 * np.log10(pwr / 1)
    return pwr_dbfs


def decimate_samples(
    samples: np.ndarray, previous_samples: np.ndarray, pc: ProcessConfig
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Returns: decimated smoothed samples, decimated sample rate and unsmoothed samples
    """

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


def rising_falling_indices(samples: np.ndarray, unsmoothed_samples: np.ndarray) -> tuple[list[int], list[int]]:
    """
    Returns: tuple of rising edge indices, falling edge indices
    """

    # Get a boolean array for all samples higher or lower than the threshold
    threshold = np.median(samples) * 2  # go just above noise floor - use smoothed samples for beep detection

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


async def async_enumerate(async_gen: AsyncIterator, start: int, count: int):
    cnt, end = start, start + count
    async for item in async_gen:
        yield cnt, item

        cnt += 1

        if cnt >= end:
            break


async def find_beep_frequencies(source_gen: AsyncIterator[np.ndarray], pc: ProcessConfig, N: int) -> list[int]:
    """
    Find beep frequencies from first N chunks.

    Returns list of unique frequencies found.
    """

    out = set()

    async for i, samples in async_enumerate(source_gen, 1, N):
        logger.debug(f"Scanning chunk no. {i}...")

        # print("find beep freq ran")
        # look for the presence of a beep within the chunk and :
        # (1) if beep found calculate the offset
        # (2) if beep not found iterate the counters and move on

        # TO DO
        # This function only to be called when --scan is given on command line
        # This function needs to operated on the first 12 chunks (3 seconds) of data
        # The detected beep array should reject any value not between 160110000 and 161120000

        # samples = await samples_queue.get()

        # f = np.linspace(sample_rate / -2, sample_rate / 2, fft_size)
        size = pc.fft_size
        step = int(size // 1.1)
        samples_to_send_to_fft = [samples[i : i + size] for i in range(0, len(samples), step)]
        num_ffts = len(samples_to_send_to_fft)  # // is an integer division which rounds down
        beep_freqs = set()

        i = 0
        for i in range(num_ffts):
            # fft = np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size]))) / fft_size
            fft = np.abs(np.fft.fftshift(np.fft.fft(samples_to_send_to_fft[i]))) / size

            # plt.plot(fft)
            # plt.show()
            # print(f"{np.max(fft)/np.median(fft)}")

            # DC Spike removal
            fft[len(fft) // 2] = np.mean(fft[(len(fft) // 2) - 10 : (len(fft) // 2) - 3])

            if (np.max(fft) / np.median(fft)) > 20:
                # if np.max(fft) > fft_thresh:
                # fft_freqs = fft_freqs_array(pc.sample_rate, size) + pc.sample_config.center_freq

                # plt.plot(fft_freqs, fft)
                # plt.show()
                # print(f"{np.median(fft)}")
                # print(f"{np.max(fft)}")

                # This needs to be changes to np.argwhere((np.max(fft) / np.median(fft)) > 20)
                # So it adds all freqs detected over the threshold
                # noise_floor = np.median(fft) * 5
                # beep_freqs.append(np.linspace(pc.sample_rate / -2, pc.sample_rate / 2, pc.fft_size)[np.argmax(fft)])

                peaks = signal.find_peaks(fft, prominence=0.3)[0]
                beep_freqs = {*beep_freqs, *fft_freqs_array(pc.sample_rate, size)[peaks]}

                # beep_freqs.append(self.sample_rate/-2+np.argmax(fft)/fft_size*self.sample_rate) more efficent??

        if len(beep_freqs) != 0:
            logger.debug(f"detected beep_freqs offsets is {beep_freqs}")
            out = {
                *out,
                *(
                    new_f
                    for f in beep_freqs
                    if 160_100_110 <= (new_f := f + pc.sample_config.center_freq) <= 161_120_000
                ),
            }

    return [int(statistics.mean(x)) for _, x in itertools.groupby(sorted(out), key=lambda f: (f + 5000) // 10000)]


async def process_sample(pc: ProcessConfig, samples_queue: asyncio.Queue, out_queue: asyncio.Queue) -> None:
    """
    Process sample asyncio task

    This task reads samples from samples_queue and detected beeps puts into out_queue.
    Runs forever till reads `None` from samples_queue.
    """

    assert pc.carrier_freq is not None, "Carrier Frequency is not set. Scan for frequencies first..."

    ch = channel(pc.carrier_freq)

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

    logger.info(f"Processing for channel={ch}/carrier_freq={pc.carrier_freq} started...")

    while True:
        samples = await samples_queue.get()

        # do we need to quit? E.g. we processed all chunks from test file
        if samples is None:
            samples_queue.task_done()
            break

        chunk_count += 1
        tot_samp_count = tot_samp_count + len(samples)

        # print(f"Received sample: {samples.size}")
        samples, sample_rate, unsmoothed_samples = decimate_samples(samples, previous_samples, pc)
        previous_samples = unsmoothed_samples

        # ... = dbc.send(samples, sample_rate)

        rising_edge_idx, falling_edge_idx = rising_falling_indices(samples, unsmoothed_samples)

        if len(rising_edge_idx) > 0 or len(falling_edge_idx) > 0:
            logger.debug(
                f"[{ch}/{pc.carrier_freq}] {chunk_count=} {rising_edge_idx=} {falling_edge_idx=} {beep_slice=}"
            )

        if len(rising_edge_idx) > 0:
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

        latitude, longitude = pc.gps_module.get_current()

        # logger.debug(f"[{pc.carrier_freq}] {rising_edge=} {falling_edge=}")

        logger.info(
            f"[{ch}/{pc.carrier_freq}] BPM: {BPM: >6.2f} | PWR: {DBFS or 0: >6.2f} dBFS | MAG: {CLIPPING: >6.3f} | BEEP_DURATION: {BEEP_DURATION: >6.4f}s | SNR: {SNR: >6.2f} | POS: {latitude} {longitude}"
        )

        #########################################################################

        res = ProcessResult(datetime.now(), ch, BPM, DBFS, CLIPPING, BEEP_DURATION, SNR, latitude, longitude)

        await out_queue.put(res)

        beep_slice = False
        stateful_index += samples.size  # + 14
        rising_edge = 0
        falling_edge = 0

        samples_queue.task_done()


async def chick_timer(
    pc: ProcessConfig,
    queue: asyncio.Queue,
    out_queue: asyncio.Queue,
):

    assert pc.carrier_freq is not None, "Carrier Frequency is not set. Scan for frequencies first..."

    async def _get_normalized_bpm(valid_bpms=(80.0, 48.0, 30.0, 20.0, 16.0)) -> tuple[float, ProcessResult]:
        process_result = await queue.get()
        normalized_BPM = min(valid_bpms, key=lambda k: abs(k - process_result.BPM))
        queue.task_done()
        return normalized_BPM, process_result

    async def _wait_for_start(start_bpm: float, snrs: list, dbfs: list) -> None:
        while True:
            normalized_bpm, res = await _get_normalized_bpm()

            if normalized_bpm == start_bpm:
                snrs.append(res.SNR)
                dbfs.append(res.DBFS)
                return

    async def _wait_specific_num_of_beeps(num: int) -> None:
        while num > 0:
            _ = await queue.get()
            queue.task_done()
            num -= 1

    async def _count_beeps_till(end_bpm: float, snrs: list, dbfs: list) -> int:
        num_beeps = 0
        digit_bpm = None

        while True:
            normalized_bpm, res = await _get_normalized_bpm()

            num_beeps += 1

            if num_beeps > 16:
                raise ChickTimerProcessingError(f"Number of detected beeps is high ({num_beeps=}).")

            match normalized_bpm:
                case x if digit_bpm is None:
                    digit_bpm = normalized_bpm
                    snrs.append(res.SNR)
                    dbfs.append(res.DBFS)

                case x if x == end_bpm:
                    if num_beeps < 2:
                        raise ChickTimerProcessingError(f"Number of detected beeps is low ({num_beeps=}).")

                    snrs.append(res.SNR)
                    dbfs.append(res.DBFS)

                    # we found `end_bpm` so end processing here
                    return num_beeps

                case _ if digit_bpm != normalized_bpm:
                    raise ChickTimerProcessingError(
                        f"Value of BPMs inside digit differs (expected={digit_bpm}/actual={normalized_bpm})."
                    )

    numbers_to_find = [
        "days_since_change_of_state",
        "days_since_hatch",
        "days_since_desertion_alert",
        "time_of_emergence",
        "weeks_batt_life_left",
        "activity_yesterday",
        "activity_two_days_ago",
        "mean_activity_last_four_days",
    ]

    cf = pc.carrier_freq
    ch = channel(cf)

    while True:
        snrs, dbfs = [], []
        out = dict.fromkeys(numbers_to_find)
        decoding_success = False
        start_dt = None

        try:

            for n in numbers_to_find:
                await _wait_for_start(20.0, snrs, dbfs)

                if start_dt is None:
                    start_dt = datetime.now()

                logger.debug(f"CT: Found Start BPM for [{n}]!")

                first_digit = await _count_beeps_till(16.0, snrs, dbfs)
                logger.debug(f"CT: [{n}] Found First digit {first_digit=}")

                second_digit = await _count_beeps_till(16.0, snrs, dbfs)
                logger.debug(f"CT: [{n}] Found Second digit {second_digit=}")

                logger.info(f"CT: Found {n}={first_digit}{second_digit}]")

                out[n] = f"{first_digit}{second_digit}"

            decoding_success = True
            logger.info(f"CT: Complete CT found! {out}")
        except ChickTimerProcessingError as err:
            logger.exception(err)

        end_dt = datetime.now()

        # snrs, dbfs contain at least one value (from _wait_for_start())
        snr_min = min(snrs)
        snr_max = max(snrs)
        snr_mean = statistics.mean(snrs)

        dbfs_min = min(dbfs)
        dbfs_max = max(dbfs)
        dbfs_mean = statistics.mean(dbfs)

        lat, lon = pc.gps_module.get_current()

        r = CTResult(
            channel=ch,
            carrier_freq=cf,
            decoding_success=decoding_success,
            start_dt=start_dt,
            end_dt=end_dt,
            snr_min=snr_min,
            snr_max=snr_max,
            snr_mean=snr_mean,
            dbfs_min=dbfs_min,
            dbfs_max=dbfs_max,
            dbfs_mean=dbfs_mean,
            lat=lat,
            lon=lon,
            **out,
        )

        await out_queue.put(r)

        if not decoding_success:
            logger.info("CT: Skiping next 100 beeps due to failed decoding!")
            await _wait_specific_num_of_beeps(100)
