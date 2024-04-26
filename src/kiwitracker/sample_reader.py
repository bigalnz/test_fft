from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, AsyncIterator, Self, TypeAlias

import numpy as np
import rtlsdr
from sqlalchemy.orm import Session

from kiwitracker.common import ProcessConfig, SampleConfig, SamplesT
from kiwitracker.db.engine import (construct_db_connection_string,
                                   construct_sqlalchemy_engine,
                                   get_sqlalchemy_engine, migrate_if_needed)
from kiwitracker.db.models import BPM, ChickTimerResult, FastTelemetryResult
from kiwitracker.exceptions import CarrierFrequencyNotFound
from kiwitracker.gps import GPSDummy, GPSReal
from kiwitracker.logging import setup_logging
from kiwitracker.sample_processor import (chick_timer, fast_telemetry,
                                          find_beep_frequencies,
                                          process_sample)

RtlSdr: TypeAlias = rtlsdr.rtlsdraio.RtlSdrAio

logger = logging.getLogger("KiwiTracker")


class SampleReader:
    sample_config: SampleConfig

    sdr: RtlSdr | None = None
    """RtlSdr instance"""

    aio_qsize: int = 100

    def __init__(self, sample_config: SampleConfig, buffer: SampleBuffer | None = None):
        self.sample_config = sample_config
        self._buffer = buffer
        self._running_sync = False
        self._running_async = False
        self.aio_queue: asyncio.Queue[SamplesT | None] = asyncio.Queue(maxsize=self.aio_qsize)
        self._read_future: asyncio.Future | None = None
        self._aio_streaming = False
        self._aio_loop: asyncio.AbstractEventLoop | None = None
        self._callback_tasks: set[asyncio.Task] = set()
        self._callback_futures: set[concurrent.futures.Future] = set()
        self._wrapped_futures: set[asyncio.Future] = set()
        self._cleanup_task: asyncio.Task | None = None

    @property
    def sample_rate(self):
        return self.sample_config.sample_rate

    @property
    def center_freq(self):
        return self.sample_config.center_freq

    @property
    def num_samples(self):
        return self.sample_config.read_size

    @property
    def gain(self):
        return self.sample_config.gain

    @property
    def gain_values_db(self) -> list[float]:
        """List of possible values for :attr:`gain` in dB
        (instead of "tenths of dB")
        """
        if self.sdr is None:
            raise RuntimeError("SampleReader not open")
        return [v / 10 for v in self.sdr.gain_values]

    @property
    def buffer(self) -> SampleBuffer | None:
        return self._buffer

    @buffer.setter
    def buffer(self, value: SampleBuffer | None):
        if value is self._buffer:
            return
        if self._aio_streaming:
            raise RuntimeError("cannot change buffer while streaming")
        self._buffer = value

    def read_samples(self) -> SamplesT:
        """Read :attr:`num_samples` from the device"""
        self._ensure_sync()
        if self.sdr is None:
            raise RuntimeError("SampleReader not open")
        samples = self.sdr.read_samples(self.num_samples)
        if TYPE_CHECKING:
            # This is just because `read_samples()`` can return a list
            # if numpy isn't installed
            assert isinstance(samples, np.ndarray)
        return samples

    async def open_stream(self):
        self._ensure_async()
        if self.num_samples % 512 != 0:
            raise ValueError("num_samples (chunk size) must be a multiple of 512")
        if self.sdr is None:
            raise RuntimeError("SampleReader not open")
        assert self._read_future is None
        loop = asyncio.get_running_loop()
        self._aio_loop = loop
        self._aio_streaming = True
        fut = loop.run_in_executor(
            None,
            self.sdr.read_samples_async,
            self._async_callback,
            self.num_samples,
        )
        self._read_future = asyncio.ensure_future(fut)
        self._cleanup_task = asyncio.create_task(self._bg_task_loop())

    async def close_stream(self):
        if not self._aio_streaming:
            return
        self._aio_streaming = False
        if self.sdr is not None:
            self.sdr.cancel_read_async()
        t = self._cleanup_task
        self._cleanup_task = None
        if t is not None:
            await t
        t = self._read_future
        self._read_future = None
        if t is not None:
            try:
                await t
            except Exception as exc:
                print(exc)
                print("fixme")
        while self.aio_queue.qsize() > 0:
            try:
                _ = self.aio_queue.get_nowait()
                self.aio_queue.task_done()
            except asyncio.QueueEmpty:
                break
        await self.aio_queue.put(None)
        self._wrap_futures()
        if len(self._wrapped_futures):
            await asyncio.gather(*self._wrapped_futures)

    def _async_callback(self, samples: SamplesT, *args):
        if not self._aio_streaming:
            return

        async def add_to_queue(samples: SamplesT):
            q = self.buffer if self.buffer is not None else self.aio_queue
            try:
                if self.buffer is None:
                    self.aio_queue.put_nowait(samples)
                else:
                    await self.buffer.put_nowait(samples)
            except asyncio.QueueFull:
                print(f"buffer overrun: {q.qsize()=}")

        if self._aio_loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(
            add_to_queue(samples),
            self._aio_loop,
        )
        self._callback_futures.add(fut)

    def _wrap_futures(self):
        fut: concurrent.futures.Future
        for fut in self._callback_futures.copy():
            self._wrapped_futures.add(asyncio.wrap_future(fut))
            self._callback_futures.discard(fut)

    async def _wait_for_futures(self, timeout: float = 0.01):
        done: set[asyncio.Future]
        done, pending = await asyncio.wait(
            self._wrapped_futures,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for fut in done:
            self._wrapped_futures.discard(fut)

    async def _bg_task_loop(self):
        while self._aio_streaming:
            self._wrap_futures()
            if len(self._wrapped_futures):
                await self._wait_for_futures()
            if not len(self._wrapped_futures) and not len(self._callback_futures):
                await asyncio.sleep(0.1)

    def _ensure_sync(self):
        if self._running_async:
            raise RuntimeError("Currently in async mode")

    def _ensure_async(self):
        if self._running_sync:
            raise RuntimeError("Currently in sync mode")

    def open(self):
        """Open the device and set all necessary parameters"""
        self._ensure_sync()
        self._running_sync = True
        self._open()

    def _open(self):
        assert self.sdr is None
        if TYPE_CHECKING:
            # NOTE: Another workaround for the above TYPE_CHECKING stuffLIN
            #       (just ignore for now)
            assert RtlSdr is not None
        sdr = self.sdr = RtlSdr()
        sdr.sample_rate = self.sample_rate
        sdr.center_freq = self.center_freq
        sdr.gain = self.gain
        if self.sample_config.bias_tee_enable:
            sdr.set_bias_tee(True)

        # NOTE: Just for debug purposes. This might help with your gain issue
        logger.info(f" RUN TIME START {datetime.now()} \n")
        logger.info(
            " ****************************************************************************************************** "
        )
        logger.info(
            f" *******          SAMPLING RATE : {sdr.sample_rate}  | CENTER FREQ: {sdr.center_freq}  | GAIN {sdr.gain}                ****** "
        )
        logger.info(
            " ******* dBFS closer to 0 is stronger ** Clipping over 0.5 is too much. Saturation at 1 *************** "
        )
        logger.info(
            " ****************************************************************************************************** "
        )
        print(f"{self.gain_values_db=}")

    def close(self):
        """Close the device if it's currently open"""
        self._ensure_sync()
        self._close()
        self._running_sync = False

    def _close(self):
        sdr = self.sdr
        if sdr is None:
            return
        self.sdr = None
        if self.sample_config.bias_tee_enable:
            sdr.set_bias_tee(False)
        sdr.close()

    async def aopen(self):
        self._ensure_async()
        self._running_async = True
        self._open()

    async def aclose(self):
        self._ensure_async()
        try:
            await self.close_stream()
        finally:
            self._close()
        self._running_async = False

    # NOTE: These two methods allow use as a context manager:
    #       with SampleReader() as reader:
    #           ...
    #
    #       This way it will always close when the program does
    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self) -> Self:
        await self.aopen()
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    async def __anext__(self) -> SamplesT:
        if not self._aio_streaming:
            raise StopAsyncIteration
        if self.buffer is not None:
            raise RuntimeError("cannot use async for if SampleBuffer is set")
        samples = await self.aio_queue.get()
        self.aio_queue.task_done()
        if samples is None:
            raise StopAsyncIteration
        return samples

    def __aiter__(self):
        if self.buffer is not None:
            raise RuntimeError("cannot use async for if SampleBuffer is set")
        return self


class SampleBuffer:
    """Buffer for samples with thread-safe reads and writes

    Behavior is similar to :class:`queue.Queue`, with the exception of
    the ``task_done()`` and ``join()`` methods (which are not present).
    """

    maxsize: int
    """The maximum length for the buffer. If less than or equal to zero, the
    buffer length is infinite
    """

    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = maxsize
        self._samples: SamplesT = np.zeros(0, dtype=np.complex128)
        self._lock: asyncio.Lock = asyncio.Lock()
        self._notify_w = asyncio.Condition(self._lock)
        self._notify_r = asyncio.Condition(self._lock)

    async def put(self, samples: SamplesT, block: bool = True, timeout: float | None = None):
        """Append new samples to the end of the buffer, blocking if necessary

        If *block* is True and *timeout* is ``None``, block if necessary until
        enough space is available to write the samples. If *timeout* is given,
        blocks at most *timeout* seconds and raises :class:`~asyncio.QueueFull`
        if no space was available during that time.

        Otherwise (if *block* is False), write the samples if enough space is
        immediately available (raising :class:`~asyncio.QueueFull` if necessary)


        Raises:
            QueueFull: If a timeout occurs waiting for available write space
        """

        async with self._lock:
            sample_size = samples.size

            def can_write():
                return len(self) < self.maxsize - sample_size

            if not can_write():
                if not block:
                    raise asyncio.QueueFull()
                if timeout is not None:
                    try:
                        async with asyncio.timeout(timeout):
                            await self._notify_w.wait_for(can_write)
                    except asyncio.TimeoutError:
                        raise asyncio.QueueFull()
                else:
                    await self._notify_w.wait_for(can_write)
            self._samples = np.concatenate((self._samples, samples))
            self._notify_r.notify_all()

    async def put_nowait(self, samples: SamplesT):
        """Equivalent to ``put(samples, block=False)``"""
        await self.put(samples, block=False)

    async def get(self, count: int, block: bool = True, timeout: float | None = None) -> SamplesT:
        """Get *count* number of samples and remove them from the buffer

        If *block* is True and *timeout* is ``None``, block if necessary for
        enough samples to exist on the buffer.  If *timeout* is given,
        blocks at most *timeout* seconds and raises :class:`~asyncio.QueueEmpty`
        if no samples were available during that time.

        Otherwise (if *block* is False), return the samples if immediately
        available (raising :class:`~asyncio.QueueEmpty`) if necessary)

        Raises:
            QueueEmpty: If a timeout occurs waiting for samples
        """

        def has_enough_samples():
            return len(self) >= count

        async with self._lock:
            if not has_enough_samples():
                if not block:
                    raise asyncio.QueueEmpty()
                if timeout is not None:
                    try:
                        async with asyncio.timeout(timeout):
                            await self._notify_w.wait_for(has_enough_samples)
                    except asyncio.TimeoutError:
                        raise asyncio.QueueEmpty()
                else:
                    await self._notify_r.wait_for(has_enough_samples)
            samples = self._samples[:count]
            self._samples = self._samples[count:]
            self._notify_w.notify_all()
            return samples

    async def get_nowait(self, count: int) -> SamplesT:
        """Equivalent to ``get(count, block=False)``"""
        return await self.get(count, block=False)

    def qsize(self) -> int:
        return len(self)

    def empty(self) -> bool:
        return len(self) == 0

    def full(self) -> bool:
        if self.maxsize > 0:
            return len(self) >= self.maxsize
        return False

    def __len__(self) -> int:
        return self._samples.size


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-f",
        "--from-file",
        dest="infile",
        help="Read samples from the given filename and process them",
    )
    p.add_argument(
        "-db",
        "--database",
        dest="db",
        default=None,
        help="SQLite database where to store processed results. Defaults to `main.db`. Environment variable KIWITRACKER_DB has priority.",
    )

    p.add_argument(
        "-d",
        "--delete-database",
        dest="deletedb",
        action="store_true",
        help="If SQLite database file exists upon start, it is deleted.",
    )

    p.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        help="Read samples from the device and save to the given filename",
    )
    p.add_argument(
        "-m",
        "--max-samples",
        dest="max_samples",
        type=int,
        help='Number of samples to read when "-o/--outfile" is specified',
    )
    p.add_argument(
        "--scan",
        dest="scan",
        action="store_true",
        help="Scan for frequencies in first 3sec",
    )
    p.add_argument(
        "--no-use-gps",
        dest="no_use_gps",
        action="store_true",
        help="Set this flag to not use GPS module",
    )

    s_group = p.add_argument_group("Sampling")
    s_group.add_argument(
        "-c",
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=SampleConfig.read_size,
        help="Chunk size for sdr.read_samples (default: %(default)s)",
    )
    s_group.add_argument(
        "-s",
        "--sample-rate",
        dest="sample_rate",
        type=float,
        default=SampleConfig.sample_rate,
        help="SDR sample rate (default: %(default)s)",
    )
    s_group.add_argument(
        "--center-freq",
        dest="center_freq",
        type=float,
        default=SampleConfig.center_freq,
        help="SDR center frequency (default: %(default)s)",
    )
    s_group.add_argument(
        "-g",
        "--gain",
        dest="gain",
        type=float,
        default=SampleConfig.gain,
        help="SDR gain (default: %(default)s)",
    )
    s_group.add_argument(
        "--bias-tee",
        dest="bias_tee",
        action="store_true",
        help="Enable bias tee",
    )

    s_group.add_argument(
        "-log", "--loglevel", default="warning", help="Provide logging level. Example --loglevel debug, default=warning"
    )

    p_group = p.add_argument_group("Processing")
    p_group.add_argument(
        "--carrier",
        dest="carrier",
        type=float,
        nargs="?",
        const=ProcessConfig.carrier_freq,
        # default=ProcessConfig.carrier_freq,
        help="Carrier frequency to process (default: %(default)s)",
    )

    args = p.parse_args()

    setup_logging(level=args.loglevel.upper())

    if args.deletedb:
        db_filename = construct_db_connection_string(db_file=args.db).removeprefix("sqlite:///")

        if os.path.exists(db_filename):
            logger.info(f"Deleting DB file {db_filename}")
            os.remove(db_filename)

    construct_sqlalchemy_engine(db_file=args.db)
    engine = get_sqlalchemy_engine()

    logger.info(f"Using DB connection URL: {engine.url.render_as_string(hide_password=False)}")

    migrate_if_needed(engine, "head")

    if args.scan and args.carrier is not None:
        print("--scan and --carrier cannot be provided simultaneously.")
        return
    elif not args.scan and args.carrier is None:
        args.carrier_freq = ProcessConfig.carrier_freq

    if args.no_use_gps:
        gps_module = GPSDummy()
    else:
        gps_module = GPSReal()

    gps_module.connect()

    sample_config = SampleConfig(
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        gain=args.gain,
        bias_tee_enable=args.bias_tee,
        read_size=args.chunk_size,
    )

    process_config = ProcessConfig(
        sample_config=sample_config,
        carrier_freq=args.carrier,
        gps_module=gps_module,
    )

    if args.infile is not None:
        # import cProfile
        # import io
        # import pstats
        # from pstats import SortKey

        # pr = cProfile.Profile()

        # pr.enable()

        process_config.running_mode = "disk"

        asyncio.run(
            pipeline(
                process_config=process_config,
                source_gen=source_file(
                    filename=args.infile,
                    N=process_config.num_samples_to_process,
                    num_chunks=None,
                ),
                task_results=results_pipeline,
            )
        )

        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

    elif args.outfile is not None:
        logger.error("Readonly mode not implemented (yet.)")
        return
    else:
        process_config.running_mode = "radio"
        asyncio.run(
            pipeline(
                process_config=process_config,
                source_gen=source_radio(
                    reader=SampleReader(sample_config),
                    buffer=SampleBuffer(maxsize=process_config.num_samples_to_process * 3),
                    num_samples_to_process=process_config.num_samples_to_process,
                ),
                task_results=results_pipeline,
            )
        )

    # asyncio.run(run_main_2(sample_config=sample_config, process_config=process_config))

    # if args.infile is not None:
    #     process_config.running_mode = "disk"

    #     run_from_disk(
    #         process_config=process_config,
    #         filename=args.infile,
    #     )
    # elif args.outfile is not None:
    #     assert args.max_samples is not None
    #     asyncio.run(
    #         run_readonly(
    #             sample_config=sample_config,
    #             filename=args.outfile,
    #             max_samples=args.max_samples,
    #         )
    #     )
    # else:
    #     asyncio.run(run_main(sample_config=sample_config, process_config=process_config))


async def store_bpm_to_db(queue: asyncio.Queue) -> None:
    with Session(get_sqlalchemy_engine()) as db_session:
        while True:
            r = await queue.get()

            bpm = BPM(
                dt=r.date,
                channel=r.channel,
                bpm=r.BPM,
                dbfs=r.DBFS,
                clipping=r.CLIPPING,
                duration=r.BEEP_DURATION,
                snr=r.SNR,
                lat=r.latitude,
                lon=r.longitude,
            )

            db_session.add(bpm)

            db_session.commit()
            queue.task_done()


async def store_ft_to_db(queue: asyncio.Queue) -> None:
    with Session(get_sqlalchemy_engine()) as db_session:
        while True:
            r = await queue.get()

            ft = FastTelemetryResult(
                channel=r.channel,
                carrier_freq=r.carrier_freq,
                start_dt=r.start_dt,
                end_dt=r.end_dt,
                snr_min=r.snr_min,
                snr_max=r.snr_max,
                snr_mean=r.snr_mean,
                dbfs_min=r.dbfs_min,
                dbfs_max=r.dbfs_max,
                dbfs_mean=r.dbfs_mean,
                lat=r.lat,
                lon=r.lon,
                mode=r.mode,
                d1=r.d1,
                d2=r.d2,
            )

            db_session.add(ft)
            db_session.commit()

            queue.task_done()


async def store_ct_to_db(queue: asyncio.Queue) -> None:
    with Session(get_sqlalchemy_engine()) as db_session:
        while True:
            r = await queue.get()

            ct = ChickTimerResult(
                channel=r.channel,
                carrier_freq=r.carrier_freq,
                decoding_success=r.decoding_success,
                start_dt=r.start_dt,
                end_dt=r.end_dt,
                snr_min=r.snr_min,
                snr_max=r.snr_max,
                snr_mean=r.snr_mean,
                dbfs_min=r.dbfs_min,
                dbfs_max=r.dbfs_max,
                dbfs_mean=r.dbfs_mean,
                lat=r.lat,
                lon=r.lon,
                days_since_change_of_state=r.days_since_change_of_state,
                days_since_hatch=r.days_since_hatch,
                days_since_desertion_alert=r.days_since_desertion_alert,
                time_of_emergence=r.time_of_emergence,
                weeks_batt_life_left=r.weeks_batt_life_left,
                activity_yesterday=r.activity_yesterday,
                activity_two_days_ago=r.activity_two_days_ago,
                mean_activity_last_four_days=r.mean_activity_last_four_days,
            )

            db_session.add(ct)
            db_session.commit()

            queue.task_done()


async def results_pipeline(
    pc: ProcessConfig,
    queue: asyncio.Queue,
) -> None:

    fast_telemetry_queue = asyncio.Queue()
    chick_timer_queue = asyncio.Queue()
    store_bpm_to_db_queue = asyncio.Queue()
    store_ct_to_db_queue = asyncio.Queue()
    store_ft_to_db_queue = asyncio.Queue()

    tasks = [
        asyncio.create_task(store_bpm_to_db(store_bpm_to_db_queue)),
        asyncio.create_task(chick_timer(pc, chick_timer_queue, [store_ct_to_db_queue])),
        asyncio.create_task(fast_telemetry(pc, fast_telemetry_queue, [store_ft_to_db_queue])),
        asyncio.create_task(store_ct_to_db(store_ct_to_db_queue)),
        asyncio.create_task(store_ft_to_db(store_ft_to_db_queue)),
    ]

    while True:
        bpm_result = await queue.get()

        for q in (store_bpm_to_db_queue, chick_timer_queue, fast_telemetry_queue):
            await q.put(bpm_result)

        queue.task_done()

    # wait for queues
    for q in (
        store_bpm_to_db_queue,
        chick_timer_queue,
        store_ct_to_db_queue,
        store_ft_to_db_queue,
        fast_telemetry_queue,
    ):
        await q.join()

    # cancel all tasks
    for t in tasks:
        t.cancel()


async def run_readonly(sample_config: SampleConfig, filename: str, max_samples: int):
    chunk_size = sample_config.read_size
    nrows = max_samples // sample_config.read_size
    if nrows * chunk_size < max_samples:
        nrows += 1
    samples = np.zeros((nrows, chunk_size), dtype=np.complex128)
    sample_config = SampleConfig(read_size=chunk_size)
    reader = SampleReader(sample_config)

    async with reader:
        await reader.open_stream()
        i = 0
        count = 0
        async for _samples in reader:
            if count == 0:
                print(f"{_samples.size=}")
            samples[i, :] = _samples
            count += _samples.size
            # print(f'{i}\t{reader.aio_queue.qsize()=}\t{count=}')
            i += 1
            if count >= max_samples:
                break
    samples = samples.flatten()[:max_samples]
    np.save(filename, samples)


def filename_to_dtype(filename):
    file_extension = os.path.splitext(filename)[1]
    file_dtype = np.complex64

    match file_extension:
        case ".fc32":
            file_dtype = np.dtype(np.complex64)
        case ".sc8":
            file_dtype = np.dtype(np.int8)
        case ".s8":
            file_dtype = np.dtype(np.uint8)
        case ".npy":
            # read the sample data type from the first sample in the file
            file_dtype = np.dtype(type(np.load(filename, mmap_mode="r")[0]))
        case _:
            raise ValueError(f"Unknown file extension {file_extension}. Rename to one of .fc32, .sc8, .s8 or .npy")

    return file_dtype


def chunk_numpy_file(filename, dtype, N):
    current_offset = 0
    while True:
        arr = np.fromfile(filename, dtype=dtype, count=N, offset=current_offset)

        # Convert unsigned 8 bit samples to 32 bit floats and complex
        # https://k3xec.com/packrat-processing-iq/ (RTL-SDR part)
        if dtype == "uint8":
            iq = arr.astype(np.float32).view(np.complex64)  # 255 + 255j   0 + 0j
            iq /= 127.5  # 2 + 2j       0 + 0j
            iq -= 1 + 1j  # 1 + 1j      -1 - 1j
            arr = iq.copy()
        elif dtype == "int8":
            iq = arr.astype(np.float32).view(np.complex64)  # 128 + 128j  -127 - 127j
            iq /= 128  # 1 + 1j      -0.992 - 0.992j
            arr = iq.copy()
        elif dtype == "complex128":
            arr = arr.astype(np.complex64)

        if len(arr) == 0:
            break

        yield arr

        current_offset += N * dtype.itemsize


async def source_file(
    filename: str,
    N: int,
    num_chunks: int | None,
) -> AsyncIterator[np.ndarray]:
    """
    filename          -> file to read from
    N                 -> from ProcessConfig.num_samples_to_process
    num_chunks        -> how many chunks to read (or None for all)
    """

    file_dtype = filename_to_dtype(filename)

    for chunks_processed, chunk in enumerate(chunk_numpy_file(filename, file_dtype, N), 1):
        yield chunk
        await asyncio.sleep(0)

        if num_chunks is not None and chunks_processed >= num_chunks:
            break


async def source_radio(
    reader: SampleReader,
    buffer: SampleBuffer,
    num_samples_to_process: int,
) -> AsyncIterator[np.ndarray]:
    """
    reader                  -> the radio
    buffer                  -> where to put samples from the radio
    num_samples_to_process  -> number of samples in one chunk
    """

    reader.buffer = buffer
    async with reader:
        await reader.open_stream()
        while True:
            chunk = await buffer.get(num_samples_to_process)
            yield chunk


async def _discard_results(
    _: ProcessConfig,
    queue: asyncio.Queue,
) -> None:
    """
    discard all results from async queue
    """

    while True:
        _ = await queue.get()
        queue.task_done()


async def scan_for_frequencies(
    source_gen: AsyncIterator[np.ndarray],
    process_config: ProcessConfig,
) -> list[float]:

    assert process_config.carrier_freq is None

    try:
        frequencies = await find_beep_frequencies(source_gen, process_config, N=13)

        if not frequencies:
            logger.error("No frequency detected, exiting...")
            raise CarrierFrequencyNotFound()
        else:
            logger.info(f"Frequencies detected: {frequencies} - end scanning...")

        return frequencies

    except CarrierFrequencyNotFound:
        logger.exception("Carrier frequency not found, interrupting sample processing...")
        raise


async def pipeline(
    process_config: list[ProcessConfig] | ProcessConfig,
    source_gen: AsyncIterator[np.ndarray],
    task_results=None,
) -> None:
    """
    process_config     -> ...
    task_samples_input -> callable with one argument (samples_queue), must return async task
    task_results       -> callable with one argument (out_queue), must return async task
                          can be None (then all results are discarded)
    """

    if task_results is None:
        task_results = _discard_results

    # what process config we have?
    # - list of process configs
    # - single process config with carrier_freq=None (should scan.)
    # - single process config with defined carrier_freq
    match process_config:
        case list():
            # TODO: process config could be list (define carrier frequencies from command line)
            # list of fully defined ProcessConfigs (with carrier_freq defined from command line)
            raise NotImplementedError("Multiple process_configs in pipeline not yet implemented.")
        case ProcessConfig() if process_config.carrier_freq is None:
            # carrier_freq is None - we should scan for frequencies
            logger.info("Carrier frequency not set - start scanning...")
            frequencies = await scan_for_frequencies(source_gen, process_config)

            # TODO: create multime process sample tasks, each with different process_config, frequency, queues...
            logger.info(f"Picking first one: {frequencies[0]}")
            process_config.carrier_freq = frequencies[0]
            process_config = [process_config]
        case ProcessConfig():
            # fully defined ProcessConfig (with carrier_freq)
            process_config = [process_config]
        case _:
            raise ValueError(f"Type of process_config {type(process_config)} not understood.")

    # for each detected/defined frequency we create two tasks:
    # - task for processing sample
    # - task for handling processed sample (detected BPM.)
    #
    # for each `processing sample task` we create `process queue` and `result_queue`
    # so we can distribute samples from task_samples_source() to each `process queue`
    # `processing sample task` will put detected BPMs to `result_queue` for further handling.
    process_queues, result_queues, process_tasks, result_tasks = set(), set(), set(), set()
    for i, pc in enumerate(process_config, 1):
        logger.debug(f"Creating sample process task and results task no.{i}, {pc.carrier_freq=}")
        result_queue = asyncio.Queue()
        process_queue = asyncio.Queue()

        task_sample_processor = asyncio.create_task(process_sample(pc, process_queue, [result_queue]))
        task_result = asyncio.create_task(task_results(pc, result_queue))

        process_queues.add(process_queue)
        result_queues.add(result_queue)

        process_tasks.add(task_sample_processor)
        result_tasks.add(task_result)

    # distribute samples to all process queues:
    async for sample in source_gen:
        for pq in process_queues:
            await pq.put(sample)

    # wait for processing:
    for pq in [*process_queues, *result_queues]:
        await pq.join()

    # cancel all tasks:
    for t in [*process_tasks, *result_tasks]:
        t.cancel()


if __name__ == "__main__":
    main()
