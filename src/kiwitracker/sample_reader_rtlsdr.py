import asyncio
import concurrent.futures
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Self, TypeAlias

import numpy as np
import rtlsdr

from kiwitracker.common import SampleConfig, SamplesT

# TYPE_CHECKING, AsyncIterator, Self, TypeAlias
# ProcessConfig, SampleConfig, SamplesT


RtlSdr: TypeAlias = rtlsdr.rtlsdraio.RtlSdrAio
logger = logging.getLogger("KiwiTracker")


class SampleReaderRtlSdr:
    sample_config: SampleConfig

    sdr: RtlSdr | None = None
    """RtlSdr instance"""

    aio_qsize: int = 100

    def __init__(self, sample_config: SampleConfig, buffer: "SampleBuffer" | None = None):
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
    def buffer(self) -> "SampleBuffer" | None:
        return self._buffer

    @buffer.setter
    def buffer(self, value: "SampleBuffer" | None):
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
