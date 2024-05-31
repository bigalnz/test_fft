import asyncio
import logging
import threading
import time

import numpy as np

from kiwitracker.common import SampleConfig

logger = logging.getLogger("KiwiTracker")


class BufferDummy:
    def __init__(self):
        self.input_queue = asyncio.Queue()  # max 3 samples in the queue
        self.samples = np.zeros(0, dtype="complex64")

    async def get(self, count):
        while True:

            tmp = []

            if self.input_queue.empty():
                await asyncio.sleep(0)
            else:
                while not self.input_queue.empty():
                    tmp.append(await self.input_queue.get())
                    self.input_queue.task_done()

            if tmp:
                self.samples = np.concatenate((self.samples, *tmp))

            if self.samples.size >= count:
                samples = self.samples[:count]
                self.samples = self.samples[count:]
                logger.debug(f"RETURNED {samples.size}")
                return samples


class SampleReaderDummy:
    def __init__(self, sample_config: SampleConfig):
        self.sc = sample_config
        self.buffer = None
        self.device_handle = None
        self.thread = None
        self.aio_loop = None

    async def open_stream(self):

        self.aio_loop = asyncio.get_running_loop()

        def run_in_thread():

            # while True:
            #     future = asyncio.run_coroutine_threadsafe(self.buffer.input_queue.put(arr.copy()), self.aio_loop)

            #     try:
            #         future.result(1)
            #     except asyncio.TimeoutError:
            #         logger.error("Timeout putting samples to queue, cancelling!")
            #         future.cancel()

            #     logger.debug(f"Added array of size={len(arr)} to to queue.")

            # tmp = []

            while True:
                arr = np.zeros(250_000 // 2, dtype="complex64")
                # tmp.append(arr)
                self.aio_loop.call_soon_threadsafe(self.buffer.input_queue.put_nowait, arr.flatten())
                time.sleep(0.01)
                logger.debug(f"Added array of size={len(arr)} to to queue.")

        self.thread = threading.Thread(
            target=run_in_thread,
            # args=(asyncio.get_running_loop(),),
        )
        self.thread.start()

        await asyncio.sleep(0.01)

    async def __aenter__(self):
        await asyncio.sleep(0.01)
        return self

    async def __aexit__(self, *args):
        pass
