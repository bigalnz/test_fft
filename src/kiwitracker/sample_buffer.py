import asyncio

import numpy as np

from kiwitracker.common import SamplesT


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
