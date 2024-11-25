from __future__ import annotations

import asyncio
import itertools
import logging
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
from typing import AsyncIterator
from collections import deque

import numpy as np
# TODO: This plotting stuff shouldn't propably be here:
from matplotlib import pyplot as plt
from scipy import signal

from kiwitracker.common import CTResult, FTResult, ProcessConfig, ProcessResult
from kiwitracker.exceptions import ChickTimerProcessingError
from kiwitracker.fasttelemetry import FT_DICT

logger = logging.getLogger("KiwiTracker")


@lru_cache(maxsize=1)
def fir() -> np.ndarray:
    return signal.firwin(501, 0.005, pass_zero=True)


@lru_cache(maxsize=5)
def time_array(num_samples_to_process: int, sample_rate: int) -> np.ndarray:
    return np.arange(num_samples_to_process) / sample_rate


@lru_cache(maxsize=5)
def phasor(num_samples_to_process: int, sample_rate: int, freq_offset: int) -> np.ndarray:
    return np.exp(2j * np.pi * time_array(num_samples_to_process, sample_rate) * freq_offset)


@lru_cache(maxsize=5)
def fft_freqs_array(sample_rate: int, fft_size: int) -> np.ndarray:
    return np.linspace(sample_rate / -2, sample_rate / 2, fft_size)


# def channel(carrier_freq: float) -> int:
#     """Channel Number from Freq"""
#     return int(round((carrier_freq - 160.12e6) / 0.01e6))


# def channel_new(carrier_freq: float) -> int:
#     """Channel Number from Freq"""
#     return int(round(abs(carrier_freq - 160.425) / 0.01))


def channel_new(carrier_freq: float) -> int:
    """Channel Number from Freq"""
    return math.floor((carrier_freq - 160.115) / 0.01)


def snr(high_samples: np.ndarray, low_samples: np.ndarray) -> float:
    noise_pwr = np.var(low_samples)
    signal_pwr = np.var(high_samples)
    snr_db = 10 * np.log10(signal_pwr / noise_pwr)
    return snr_db


def clipping(high_samples: np.ndarray) -> float:
    clipping_max = np.max(high_samples)
    return clipping_max


def dBFS(high_samples: np.ndarray) -> float:
    pwr = np.square(high_samples)
    pwr = np.average(pwr)
    pwr_dbfs = 10 * np.log10(pwr / 1)
    return pwr_dbfs


def noise_floor(tk: np.ndarray) -> float:
     return np.mean(np.abs(tk))


def decimate_samples(
    samples: np.ndarray, previous_samples: np.ndarray, pc: ProcessConfig
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Returns: decimated smoothed samples, decimated sample rate and unsmoothed samples
    """

    samples = samples * phasor(pc.num_samples_to_process, pc.sample_rate, pc.freq_offset)[: samples.size]
    # next two lines are band pass filter?
    samples = signal.convolve(samples, fir(), mode="same", method="direct")
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
            fft[(len(fft) // 2 - 60) : (len(fft) // 2 + 60)] = np.mean(fft[(len(fft) // 2) - 80 : (len(fft) // 2) - 60])

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

                peaks = signal.find_peaks(fft, prominence=0.0015)[0]
                beep_freqs = {*beep_freqs, *fft_freqs_array(pc.sample_rate, size)[peaks]}

                # beep_freqs.append(self.sample_rate/-2+np.argmax(fft)/fft_size*self.sample_rate) more efficent??

        if len(beep_freqs) != 0:
            # beep_freqs = {f for f in beep_freqs if abs(f) <= 6_000}
            logger.debug(f"detected beep_freqs offsets is {beep_freqs}")
            out = {
                *out,
                *(
                    new_f
                    for f in beep_freqs
                    if 160_115_000 <= (new_f := f + pc.sample_config.center_freq) <= 161_115_000
                ),
            }

    grouped = [int(statistics.mean(x)) for _, x in itertools.groupby(sorted(out), key=lambda f: (f + 5000) // 10000)]

    logger.debug(f"Final grouped frequencies: {grouped}")

    return grouped


def index_of(arr: np.array, v):
    return next((idx[0] for idx, val in np.ndenumerate(arr) if val == v), -1)


# kiwitracker --center 160425000 --scan 0 -s 768000 --no-use-gps --loglevel debug -f data/sept24.fc32

# new method according:
# https://github.com/bigalnz/test_fft/blob/23-fft-channelizer/notebooks/fft_perf.ipynb

# command:
# kiwitracker --center 160425000 -f data/6_channels_768_samp.fc32 --scan 0 -s 7680000 --no-use-gps --loglevel info
async def process_sample_new(
    pc: ProcessConfig,
    samples_queue: asyncio.Queue,
    queue_output: asyncio.Queue,
) -> None:

    logger.info(
        f"process_sample_new(): started with parameters {pc.sample_config.center_freq=} {pc.sample_config.sample_rate=}"
    )

    freqs_to_discard = (pc.sample_config.center_freq - 10_000) / 1e6, (pc.sample_config.center_freq + 10_000) / 1e6

    logger.debug(
        f"process_sample_new(): will discard frequencies {freqs_to_discard=}"
    )


    def db(x):
        return 10*np.log10(x)


    # f0 = 160.425e6  # Center frequency
    f0 = int(pc.sample_config.center_freq)
    # Fs = 768e3  # Sampling rate
    Fs = int(pc.sample_config.sample_rate)

    N_fft = 1024  # Number of FFT channels
    N_time_PSD = 250
    # threshold = 2.2

    # Generate array of channel frequencies
    f = (np.fft.fftshift(np.fft.fftfreq(N_fft, 1 / Fs)) + f0) / 1e6

    # Time tag each sample
    # t = np.arange(pc.num_samples_to_process) / Fs

    cnt = 0
    prev_rising_edge_indices = {}
    noise_floors_per_channel = {}

    stored_D = deque(maxlen=2)

    while True:
        samples = await samples_queue.get()

        # do we need to quit? E.g. we processed all chunks from test file
        if samples is None:
            samples_queue.task_done()
            break

        # Reshape so we can do an FFT over an axis
        d_fft = samples.reshape((-1, N_fft))
        D = np.fft.fftshift(np.fft.fft(d_fft, axis=1), axes=(1,))

        stored_D.append(D)

        # is this first sample processed?
        if len(stored_D) == 1:
            # yes, continue reading from radio/disk
            continue

        # Time tag each sample coming from a channel
        # T = np.arange(len(D)) / Fs * N_fft

        # Now convert into power spectral density
        # 1. Reshape to (N_timestep, N_int_per_timestep, N_fft)
        # 2. Square
        # 3. Sum over N_int_per_timestep axis
        # PSD = (np.abs(D.reshape((N_time_PSD, -1, N_fft))) ** 2).mean(axis=1)
        PSD = (np.abs(stored_D[0].reshape((N_time_PSD, -1, N_fft))) ** 2).mean(axis=1)

        # Create overall spectrum
        spec = PSD.mean(axis=0)

        # Find peaks (note: I hand-tuned prominence)
        # p = signal.find_peaks(spec, prominence=0.000010)[0]
        p, prom_data = signal.find_peaks(spec, prominence=0.000010)
        prom = prom_data['prominences']

        # REMOVE PEAKS OUT OF BAND 160.110 - 161.120
        original_indexes = np.arange(len(p))
        pp = p
        p = p[(p >= np.abs(f-160.110).argmin()) & (p <= np.abs(f-161.120).argmin())]
        removed_indexes = original_indexes[~((pp >= np.abs(f-160.110).argmin()) & (pp <= np.abs(f-161.120).argmin()))]
        prom = np.delete(prom, removed_indexes)

        # MAX PEAKS
        # set how many maximum peaks to process, either 20 or len(p) which ever is smaller
        num_of_peaks = min(20, len(p))
        max_peaks = np.argpartition(prom, -num_of_peaks)[-num_of_peaks:] #get top 20 or len(p) peaks


        # filter `max_peaks`:
        # 1.) if for the channel number there are more than one frequency, discard all frequencies for that channel
        # 2.) any channel less than 0 or any channel that is over 99 - discard

        grouped_peaks = {}
        for peak in max_peaks:
            channel_no = channel_new(f[p[peak]])
            grouped_peaks.setdefault(channel_no, []).append(peak)
        max_peaks = [v[0] for k, v in grouped_peaks.items() if len(v) == 1 and (0 <= k <= 99)]

        # Extract the time series for each channel identified
        #t_kiwis = [D[:, idx] for idx in p]
        # t_kiwis = [D[:, idx] for idx in p[max_peaks]] # use this one for max peaks only
        # t_kiwis = [stored_D[0][:, idx] for idx in p[max_peaks]]

        # And extract the carrier frequencies
        #f_kiwis = f[p]
        # f_kiwis = f[p[max_peaks]]
        
        # AVERAGED PSD
        """ plt.figure(figsize=(24,8))
        plt.plot(f, db(spec))
        plt.scatter(f[p], db(spec)[p], marker='x', color='#cc0000')
        # plot the max n peaks with a 'o' symbol
        plt.scatter( f[p[max_peaks]] , db(spec)[p][max_peaks], marker='o', color='g' )
        plt.axvline(x = 160.559, color = 'g', label = 'axvline - full height')
        plt.axvline(x = 160.377, color = 'g', label = 'axvline - full height')
        plt.axvline(x = 160.4595, color = 'g', label = 'axvline - full height')
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Power dB(counts)")
        plt.show() """

        for channel_idx in p[max_peaks]:

            channel_str = f'{f[channel_idx]}'
            channel_no = channel_new(f[channel_idx])

            # append the extra samples from correct channel in future
            tk = np.append(stored_D[0][:, channel_idx], stored_D[1][:, channel_idx][:12])
            # tk = np.append(tk, stored_D[0][:12])
            #tk = np.append(tk, stored_D[0][:12, channel_idx])

            # match filtered signal
            tk = signal.convolve(np.abs(tk), [1]*13, 'valid')
    
            # discard all frequencies +/- 10kHz from center frequency
            if freqs_to_discard[0] <= f[channel_idx] <= freqs_to_discard[1]:
                continue

            if channel_no not in noise_floors_per_channel:
                nf = deque(maxlen=3)
                noise_floors_per_channel[channel_no] = nf
            else:
                nf = noise_floors_per_channel[channel_no]

            nf.append(noise_floor(tk))
            threshold = (sum(nf) / len(nf)) * 1.5

            """ logger.debug(
                f"[{channel_no:>3}/{channel_str:>10}] Computed {threshold=}"
            ) """
            
            high_samples = np.abs(tk) >= threshold
            
            rising_edge_idx = index_of(high_samples, True)
            if rising_edge_idx == -1:
                # no rising index in array, continue
                continue

            if channel_no not in prev_rising_edge_indices:
                prev_rising_edge_indices[channel_no] = (rising_edge_idx, cnt)
                continue

            falling_edge_idx = 250 - index_of(high_samples[::-1], True)

            prev = prev_rising_edge_indices.get(channel_no)
            assert prev is not None, "Previous rising index not found"

            # test if current beep is "sliced"
            # "sliced" means that rising index in current sample is 0 AND
            # the length of the beep is less than 13 (out of 250)
            if rising_edge_idx == 0:
                if falling_edge_idx < 13:
                    # we are in slice!
                    # nothing to be done here, just read next sample
                    continue

                if falling_edge_idx > 15:
                    # bogus beep? bad signal?
                    continue

            # at this point, we know we're not in "sliced" beep and we have 
            # rising_edge_idx and falling_edge_idx computed

            # test_dBFS=dBFS(np.abs(tk[rising_edge_idx:falling_edge_idx]))
            # print(test_dBFS)

            bpm = 60.0 / (((rising_edge_idx + (250 * cnt)) - (prev[0] + 250 * prev[1])) / 750.0)

            # can look to add SNR back in future release
            # snr = snr(high_samples, low_samples)
            # snr = snr(tk[np.abs(tk) >= thres], tk[np.abs(tk) < thres] )

            if not (10.0 <= bpm <= 85.0):
                logger.debug(f'[{channel_no:>3}/{channel_str:>10}] BPM: {bpm:<3.2f} not in interval from 10.0 to 85.0')
            else:
                latitude, longitude = pc.gps_module.get_current()
                res = ProcessResult(
                    date=datetime.now(),
                    channel=channel_no,
                    carrier_freq=f[channel_idx],
                    BPM=bpm,
                    DBFS=dBFS(np.abs(tk[rising_edge_idx:falling_edge_idx])),
                    CLIPPING=-1,
                    BEEP_DURATION=-1,
                    SNR=-1,
                    latitude=latitude,
                    longitude=longitude,
                )

                logger.info(
                    f"[{channel_no:>3}/{channel_str:>10}] BPM: {bpm:<3.2f} | POS: {latitude} {longitude} | dBFS: {res.DBFS:0.0f} | {rising_edge_idx=}/{cnt=}/{prev=}"
                )

                await queue_output.put(res)

            prev_rising_edge_indices[channel_no] = (rising_edge_idx, cnt)

        # increase counter to correctly compute BPMs
        cnt += 1

        samples_queue.task_done()

def is_bpm_valid(normalized_bpm: float, real_bpm: float, tolerance: float = 0.5) -> bool:
    match normalized_bpm:
        case 80.0:
            return (67.42 - tolerance) <= real_bpm <= (80 + tolerance)
        case 48.0:
            return (43.17 - tolerance) <= real_bpm <= (48.0 + tolerance)
        case 30.0:
            return (28.04 - tolerance) <= real_bpm <= (30.0 + tolerance)
        case 20.0:
            return (19.11 - tolerance) <= real_bpm <= (20.0 + tolerance)
        case 16.0:
            return (15.42 - tolerance) <= real_bpm <= (16.0 + tolerance)
        case _:
            raise ValueError(f"Unknown normalized BPM value: {normalized_bpm=:.2f}/{real_bpm=:.2f}")


def normalize_bpm(bpm: float, valid_bpms=(80.0, 48.0, 30.0, 20.0, 16.0)) -> float:
    normalized_bpm = min(valid_bpms, key=lambda k: abs(k - bpm))

    if not is_bpm_valid(normalized_bpm, bpm):
        raise ValueError(f"BPM not in valid range: {normalized_bpm=:.2f}/{bpm=:.2f}")

    return normalized_bpm


async def fast_telemetry(
    pc: ProcessConfig,
    queue: asyncio.Queue,
    out_queues: list[asyncio.Queue],
):
    assert pc.carrier_freq is not None, "Carrier Frequency is not set. Scan for frequencies first..."

    async def _wait_for_state() -> tuple[str | int, float, float]:
        while True:
            process_result = await queue.get()
            queue.task_done()

            try:
                normalized_bpm = normalize_bpm(process_result.BPM)
            except ValueError as err:
                logger.debug(
                    f"[{channel_new(pc.carrier_freq)}/{pc.carrier_freq}] fast_telemetry._wait_for_state(): {err}"
                )
                continue

            k, v = min(FT_DICT[normalized_bpm].items(), key=lambda t: abs(t[1] - process_result.BPM))
            logger.debug(f"FT: [{ch}/{cf}] partial state found (from {process_result.BPM:.2f}) {k=}/{v=}")

            return k, process_result.SNR, process_result.DBFS

    cf = pc.carrier_freq
    ch = channel_new(cf)

    while True:
        start_dt, end_dt = None, None
        snrs, dbfs = [], []

        # wait for first string from FT ("Mortality"/"Nesting"/"Not Nesting"/"Hatch"/"Deserting")
        while True:
            mode, s, d = await _wait_for_state()
            if isinstance(mode, str):
                snrs.append(s)
                dbfs.append(d)
                logger.debug(f"FT: [{ch}/{cf}] (Waiting first string) String found {mode=}")
                break

        start_dt = datetime.now()

        # find XX/YY digits - if it finds string instead of digit
        # abort process and continue from beginning

        try:
            digits = []
            for _ in range(4):
                digit, s, d = await _wait_for_state()
                if isinstance(digit, int):
                    digits.append(digit)
                    snrs.append(s)
                    dbfs.append(d)
                    logger.debug(f"FT: [{ch}/{cf}] (Waiting for 4 digits) Digits so far: {digits=}")
                else:
                    raise ValueError(f"FT: [{ch}/{cf}] Number was expected but {digit} was received.")
        except ValueError as err:
            logger.exception(err)
            continue

        end_dt = datetime.now()

        lat, lon = pc.gps_module.get_current()

        # snrs, dbfs contain at least one value (from _wait_for_start())
        snr_min = min(snrs)
        snr_max = max(snrs)
        snr_mean = statistics.mean(snrs)

        dbfs_min = min(dbfs)
        dbfs_max = max(dbfs)
        dbfs_mean = statistics.mean(dbfs)

        d1 = int(f"{digits[0]}{digits[1]}")
        d2 = int(f"{digits[2]}{digits[3]}")

        logger.info(f"FT: [{ch}/{cf}] state found {mode}-{d1}-{d2}")

        r = FTResult(
            channel=ch,
            carrier_freq=cf,
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
            mode=mode,
            d1=d1,
            d2=d2,
        )

        for q in out_queues:
            await q.put(r)


async def chick_timer(
    pc: ProcessConfig,
    queue: asyncio.Queue,
    out_queues: list[asyncio.Queue],
):

    assert pc.carrier_freq is not None, "Carrier Frequency is not set. Scan for frequencies first..."

    async def _get_normalized_bpm() -> tuple[float, ProcessResult]:
        process_result = await queue.get()
        queue.task_done()
        return normalize_bpm(process_result.BPM), process_result

    async def _wait_for_start(start_bpm: float, snrs: list, dbfs: list) -> None:
        while True:
            try:
                normalized_bpm, res = await _get_normalized_bpm()

                if normalized_bpm == start_bpm:
                    snrs.append(res.SNR)
                    dbfs.append(res.DBFS)
                    return

            except ValueError as err:
                # logger.exception(err)
                logger.debug(f"[{channel_new(pc.carrier_freq)}/{pc.carrier_freq}] chick_timer._wait_for_start(): {err}")

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
    ch = channel_new(cf)

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
        except ValueError as err:
            logger.debug(f"[{channel_new(pc.carrier_freq)}/{pc.carrier_freq}] chick_timer: {err}")
            # logger.exception(err)

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

        for q in out_queues:
            await q.put(r)

        if not decoding_success:
            logger.info("CT: Skiping next 100 beeps due to failed decoding!")
            await _wait_specific_num_of_beeps(100)
