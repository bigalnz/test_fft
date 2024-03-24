from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Self, TypeAlias

import numpy as np
import rtlsdr

RtlSdr: TypeAlias = rtlsdr.rtlsdraio.RtlSdrAio

import time

from kiwitracker.common import ProcessConfig, SampleConfig, SamplesT
from kiwitracker.sample_processor import SampleProcessor


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
        self.logger = logging.getLogger("KiwiTracker")

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
            # NOTE: Another workaround for the above TYPE_CHECKING stuff
            #       (just ignore for now)
            assert RtlSdr is not None
        sdr = self.sdr = RtlSdr()
        sdr.sample_rate = self.sample_rate
        sdr.center_freq = self.center_freq
        sdr.gain = self.gain
        if self.sample_config.bias_tee_enable:
            sdr.set_bias_tee(True)

        # NOTE: Just for debug purposes. This might help with your gain issue
        self.logger.info(f" RUN TIME START {datetime.now()} \n")
        self.logger.info(
            f" ****************************************************************************************************** "
        )
        self.logger.info(
            f" *******          SAMPLING RATE : {sdr.sample_rate}  | CENTER FREQ: {sdr.center_freq}  | GAIN {sdr.gain}                ****** "
        )
        self.logger.info(
            f" ******* dBFS closer to 0 is stronger ** Clipping over 0.5 is too much. Saturation at 1 *************** "
        )
        self.logger.info(
            f" ****************************************************************************************************** "
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

    # Setup Logging
    # create logger with 'my_application'
    logger = logging.getLogger("KiwiTracker")
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    print(os.getcwd())
    fh = logging.FileHandler("kiwitracker.log")
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    p = argparse.ArgumentParser()
    p.add_argument(
        "-f",
        "--from-file",
        dest="infile",
        help="Read samples from the given filename and process them",
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

    p_group = p.add_argument_group("Processing")
    p_group.add_argument(
        "--carrier",
        dest="carrier",
        type=float,
        default=ProcessConfig.carrier_freq,
        help="Carrier frequency to process (default: %(default)s)",
    )

    args = p.parse_args()

    sample_config = SampleConfig(
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        gain=args.gain,
        bias_tee_enable=args.bias_tee,
        read_size=args.chunk_size,
    )

    process_config = ProcessConfig(sample_config=sample_config, carrier_freq=args.carrier)

    print(sample_config)
    print(process_config)
    return

    if args.infile is not None:
        process_config.running_mode = "disk"

        out_queue = asyncio.Queue()

        asyncio.run(
            run_from_disk_2(
                process_config=process_config,
                filename=args.infile,
                out_queue=out_queue,
            )
        )

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
            file_dtype = np.dtype(np.uint8)
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

        if len(arr) == 0:
            break

        yield arr

        current_offset += N * dtype.itemsize


async def run_from_disk_2(process_config, filename, out_queue, num_chunks=None):
    file_dtype = filename_to_dtype(filename)

    processor = SampleProcessor(process_config)

    samples_queue = asyncio.Queue()

    sample_processor_task = asyncio.Task(processor.process_2(process_config, samples_queue, out_queue))

    start_time = time.time()

    N = process_config.num_samples_to_process

    for chunks_processed, chunk in enumerate(chunk_numpy_file(filename, file_dtype, N), 1):
        await samples_queue.put(chunk)

        if num_chunks is not None and chunks_processed >= num_chunks:
            break

    await samples_queue.put(None)  # Indicate that we are done with samples
    await samples_queue.join()

    finish_time = time.time()
    print(f" run time is {finish_time-start_time:.2f}")

    del sample_processor_task


# new from_disk() head:
# 2024-03-17 22:43:10,575 - KiwiTracker - INFO -  BPM :  186.01 | PWR : -7.89 dBFS | MAG :  0.451 | BEEP_DURATION :  0.1102s | SNR : -14.36 | POS : -36.8807 174.924
# 2024-03-17 22:43:10,682 - KiwiTracker - INFO -  BPM :  30.00 | PWR : -7.65 dBFS | MAG :  0.452 | BEEP_DURATION :  0.1103s | SNR :  12.01 | POS : -36.8807 174.924
# 2024-03-17 22:43:10,856 - KiwiTracker - INFO -  BPM :  17.32 | PWR : -8.24 dBFS | MAG :  0.462 | BEEP_DURATION :  0.0896s | SNR :  35.31 | POS : -36.8807 174.924
# 2024-03-17 22:43:10,943 - KiwiTracker - INFO -  BPM :  30.00 | PWR : -10.62 dBFS | MAG :  0.447 | BEEP_DURATION :  0.0896s | SNR :  37.54 | POS : -36.8807 174.924
# 2024-03-17 22:43:11,038 - KiwiTracker - INFO -  BPM :  30.33 | PWR : -11.78 dBFS | MAG :  0.288 | BEEP_DURATION :  0.0033s | SNR : -21.53 | POS : -36.8807 174.924


# old from_disk() head:
# 2024-03-17 22:41:50,581 - KiwiTracker - INFO -  BPM :  186.01 | PWR : -7.89 dBFS | MAG :  0.451 | BEEP_DURATION :  0.1102s | SNR : -14.36 | POS : -36.8807 174.924
# 2024-03-17 22:41:50,680 - KiwiTracker - INFO -  BPM :  30.00 | PWR : -7.65 dBFS | MAG :  0.452 | BEEP_DURATION :  0.1103s | SNR :  12.01 | POS : -36.8807 174.924
# 2024-03-17 22:41:50,877 - KiwiTracker - INFO -  BPM :  17.32 | PWR : -8.24 dBFS | MAG :  0.462 | BEEP_DURATION :  0.0896s | SNR :  35.31 | POS : -36.8807 174.924
# 2024-03-17 22:41:50,994 - KiwiTracker - INFO -  BPM :  30.00 | PWR : -10.62 dBFS | MAG :  0.447 | BEEP_DURATION :  0.0896s | SNR :  37.54 | POS : -36.8807 174.924
# 2024-03-17 22:41:51,101 - KiwiTracker - INFO -  BPM :  30.33 | PWR : -11.78 dBFS | MAG :  0.288 | BEEP_DURATION :  0.0033s | SNR : -21.53 | POS : -36.8807 174.924


def run_from_disk(process_config: ProcessConfig, filename: str):
    file_dtype = filename_to_dtype(filename)

    print(f"Loading samples from {filename} with data type {file_dtype}")
    samples = np.fromfile(filename, dtype=file_dtype)
    print(f"{len(samples)} samples loaded")

    # Convert unsigned 8 bit samples to 32 bit floats and complex
    if file_dtype is np.uint8:
        iq = samples.astype(np.float32).view(np.complex64)
        iq /= 127.5
        iq -= 1 + 1j
        samples = iq.copy()

    if file_dtype == np.complex128:
        samples = samples.astype(np.complex64)

    processor = SampleProcessor(process_config)

    # Organize input samples into chunks, resize handles padding the end with zeroes
    # Better would be to only read single chunks from the file
    chunk_len = processor.num_samples_to_process
    n_chunks = np.floor(samples.size / chunk_len).astype(int)
    samples.resize(n_chunks, chunk_len)

    start_time = time.time()

    for chunk in samples:
        processor.process(chunk)

    finish_time = time.time()
    print(f" run time is {finish_time-start_time:.2f}")


async def run_main(sample_config: SampleConfig, process_config: ProcessConfig):
    reader = SampleReader(sample_config)
    processor = SampleProcessor(process_config)
    buffer = SampleBuffer(maxsize=processor.num_samples_to_process * 3)
    reader.buffer = buffer

    async with reader:
        await reader.open_stream()
        while True:
            samples = await buffer.get(processor.num_samples_to_process)
            # start_time = time.time()
            await asyncio.to_thread(processor.process, samples)
            # finish_time = time.time() - start_time
            # print(f" prcoessor took : {finish_time}")


if __name__ == "__main__":
    main()
