import asyncio

import numpy as np

import kiwitracker.airspyhf as airspy
from kiwitracker.common import SampleConfig


class BufferAirspy:
    def __init__(self):
        self.input_queue = asyncio.Queue()
        self.samples = np.zeros(0, dtype="complex64")

    async def get(self, count):

        while True:
            while not self.input_queue.empty():
                samples = await self.input_queue.get()
                self.samples = np.concatenate((self.samples, samples))
                self.input_queue.task_done()

            if self._samples.size >= count:
                samples = self.samples[:count]
                self.samples = self.samples[count:]
                return samples


class SampleReaderAirspy:
    def __init__(self, sample_config: SampleConfig, loop: asyncio.AbstractEventLoop):
        self.sc = sample_config
        self.buffer = None
        self.device_handle = None
        self.aio_loop = loop

    async def open_stream(self):
        await asyncio.sleep(0.01)

    async def __aenter__(self):

        def sync_with_main_thread(complex_data):
            self.aio_loop.call_soon_threadsafe(self.buffer.input_queue.put_nowait, complex_data.copy())

        sn = airspy.get_serial()
        self.device_handle = airspy.open_device(sn)
        rates = airspy.get_sample_rates(self.device_handle)
        airspy.set_freq(self.device_handle, self.sc.center_freq)
        airspy.set_sample_rate(self.device_handle, rates[0])

        # assert self.device_handle is not None
        # assert self.buffer is not None
        # assert self.aio_loop is not None

        airspy.start_sampling(self.device_handle, sync_with_main_thread)

        await asyncio.sleep(0.01)

        return self

    async def __aexit__(self, *args):
        pass
