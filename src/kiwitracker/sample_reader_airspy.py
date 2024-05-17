import asyncio
import logging
import threading
import time

import numpy as np

import kiwitracker.airspyhf as airspy
from kiwitracker.common import SampleConfig

logger = logging.getLogger("KiwiTracker")


class BufferAirspy:
    def __init__(self):
        self.input_queue = asyncio.Queue()
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
                logger.debug(f"Returned array of size {samples.size} from BufferAirspy.get()")
                return samples


class SampleReaderAirspy:
    def __init__(self, sample_config: SampleConfig):
        self.sc = sample_config
        self.buffer = None
        self.device_handle = None
        self.aio_loop = None

    async def open_stream(self):
        self.aio_loop = asyncio.get_running_loop()

        await asyncio.sleep(0.01)

    async def __aenter__(self):

        def sync_with_main_thread(complex_data):
            logger.debug(f"{complex_data} received!")

            future = asyncio.run_coroutine_threadsafe(self.buffer.input_queue.put(complex_data.ravel()), self.aio_loop)

            try:
                future.result(1)
            except asyncio.TimeoutError:
                logger.error("Timeout putting samples to queue, cancelling!")
                future.cancel()

        sn = airspy.get_serial()
        logger.debug(f"Serial Number: {sn}")
        self.device_handle = airspy.open_device(sn)
        logger.debug(f"Device Handle: {self.device_handle}")
        rates = airspy.get_sample_rates(self.device_handle)
        logger.debug(f"Rates: {rates}")
        status = airspy.set_freq(self.device_handle, self.sc.center_freq)
        logger.debug(f"Frequency {self.sc.center_freq} set, {status=}")
        status = airspy.set_sample_rate(self.device_handle, rates[0])
        logger.debug(f"Sample rate {rates[0]} set, {status=}")

        airspy.set_default_options(self.device_handle)

        airspy.start_sampling(self.device_handle, sync_with_main_thread)

        logger.debug(f"Sampling started!")

        await asyncio.sleep(0.01)

        return self

    async def __aexit__(self, *args):
        pass
