from typing import Self, TYPE_CHECKING
import argparse
import threading
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from rtlsdr import *
from pylab import *
import time
from scipy import signal

if TYPE_CHECKING:
    # NOTE: This is just for static type checking because rtlsdr.RtlSdr isn't
    #       really set up properly.. I should probably fix that
    from rtlsdr.rtlsdr import RtlSdr


SamplesT = npt.NDArray[np.complex128]
"""Alias for sample arrays"""



class QueueEmpty(Exception):
    """Raised by SampleBuffer.get/get_nowait"""

class QueueFull(Exception):
    """Raised by SampleBuffer.put/put_nowait"""


class SampleReader:
    sample_rate: float = 2.4e6
    center_freq: float = 160270968

    gain: float|str = 44.5
    """gain in dB"""

    num_samples: int = 16384
    """Number of samples to read in each iteration"""

    sdr: RtlSdr|None = None
    """RtlSdr instance"""

    @property
    def gain_values_db(self) -> list[float]:
        """List of possible values for :attr:`gain` in dB
        (instead of "tenths of dB")
        """
        if self.sdr is None:
            raise RuntimeError('SampleReader not open')
        return [v / 10 for v in self.sdr.gain_values]

    def read_samples(self) -> SamplesT:
        """Read :attr:`num_samples` from the device
        """
        if self.sdr is None:
            raise RuntimeError('SampleReader not open')
        samples = self.sdr.read_samples(self.num_samples)
        if TYPE_CHECKING:
            # This is just because `read_samples()`` can return a list
            # if numpy isn't installed
            assert isinstance(samples, np.ndarray)
        return samples

    def open(self):
        """Open the device and set all necessary parameters
        """
        assert self.sdr is None
        if TYPE_CHECKING:
            # NOTE: Another workaround for the above TYPE_CHECKING stuff
            #       (just ignore for now)
            assert RtlSdr is not None
        sdr = self.sdr = RtlSdr()
        sdr.sample_rate = self.sample_rate
        sdr.center_freq = self.center_freq
        sdr.gain = self.gain

        # Now we should read the *actual* values back from the device
        self.sample_rate = sdr.sample_rate
        self.center_freq = sdr.center_freq
        self.gain = sdr.gain

        # NOTE: Just for debug purposes. This might help with your gain issue
        print(f'{sdr.sample_rate=}, {sdr.center_freq=}, {sdr.gain=}')
        print(f'{self.gain_values_db=}')

    def close(self):
        """Close the device if it's currently open
        """
        sdr = self.sdr
        if sdr is None:
            return
        self.sdr = None
        sdr.close()

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
        self._lock: threading.RLock = threading.RLock()
        self._notify_w = threading.Condition(self._lock)
        self._notify_r = threading.Condition(self._lock)

    def put(self, samples: SamplesT, timeout: float|None = None):
        """Append new samples to the end of the buffer, blocking if necessary

        If *timeout* is given, waits for at most *timeout* seconds for enough
        available room on the buffer to write the samples.
        Otherwise, waits indefinitely.

        Raises:
            QueueFull: If a timeout occurs waiting for available write space
        """
        with self._lock:
            new_size = len(self) + samples.size
            if self.maxsize > 0 and new_size > self.maxsize:
                def can_write():
                    return new_size <= self.maxsize
                r = self._notify_w.wait_for(can_write, timeout=timeout)
                if not r:
                    raise QueueFull()
            self._samples = np.concatenate((self._samples, samples))
            self._notify_r.notify_all()

    def put_nowait(self, samples: SamplesT):
        """Immediately append new samples to the end of the buffer

        Raises:
            QueueFull: If not enough write space is available
        """
        with self._lock:
            new_size = len(self) + samples.size
            if self.maxsize > 0 and new_size > self.maxsize:
                raise QueueFull()
            self._samples = np.concatenate((self._samples, samples))
            self._notify_r.notify_all()

    def get(self, count: int, timeout: float|None = None) -> SamplesT:
        """Get *count* number of samples and remove them from the buffer

        If *timeout* is given, waits at most *timeout* seconds for enough
        samples to be written. Otherwise, waits indefinitely.

        Raises:
            QueueEmpty: If a timeout occurs waiting for samples
        """
        def has_enough_samples():
            return len(self) >= count

        with self._lock:
            if not has_enough_samples():
                r = self._notify_r.wait_for(has_enough_samples, timeout=timeout)
                if not r:
                    raise QueueEmpty()
            samples = self._samples[:count]
            self._samples = self._samples[count:]
            self._notify_w.notify_all()
            return samples

    def get_nowait(self, count: int) -> SamplesT:
        """Get *count* number of samples and remove them from the buffer

        Raises:
            QueueEmpty: If there aren't enough samples
        """
        with self._lock:
            if len(self) < count:
                raise QueueEmpty()
            samples = self._samples[:count]
            self._samples = self._samples[count:]
            self._notify_w.notify_all()
            return samples

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


class SampleProcessor:
    threshold: float = 0.9
    freq_offset: float = -43.8e4 #Hz
    beep_duration: float = 0.017 # seconds

    num_samples_to_process: int = int(1.024e6)
    """Number of samples needed to process"""

    sample_rate: float
    stateful_index: int

    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = sample_rate
        self.stateful_index = 0

    @property
    def fft_size(self) -> int:
        # this makes sure there's at least 1 full chunk within each beep
        return int(self.beep_duration * self.sample_rate / 2)

    def process_from_buffer(self, buffer: SampleBuffer) -> SamplesT:
        """Wait for enough samples on the buffer, then process them
        """
        samples = buffer.get(self.num_samples_to_process)
        self.process(samples)
        return samples

    def process(self, samples: SamplesT):
        fft_size = self.fft_size
        f = np.linspace(self.sample_rate/-2, self.sample_rate/2, fft_size)
        num_ffts = len(samples) // fft_size # // is an integer division which rounds down
        fft_thresh = 0.1
        beep_freqs = []
        for i in range(num_ffts):
            fft = np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size]))) / fft_size
            if np.max(fft) > fft_thresh:
                beep_freqs.append(np.linspace(self.sample_rate/-2, self.sample_rate/2, fft_size)[np.argmax(fft)])
            plt.plot(f,fft)
        #print(beep_freqs)
        #plt.show()

        t = np.arange(len(samples))/self.sample_rate
        samples = samples * np.exp(2j*np.pi*t*self.freq_offset)
        h = signal.firwin(501, 0.02, pass_zero=True)
        samples = np.convolve(samples, h, 'valid')
        samples = samples[::100]
        sample_rate = self.sample_rate/100
        samples = np.abs(samples)
        samples = np.convolve(samples, [1]*10, 'valid')/10
        max_samp = np.max(samples)
        # samples /= np.max(samples)
        #print(f"max sample : {max_samp}")
        #plt.plot(samples)
        #plt.show()

        # Get a boolean array for all samples higher or lower than the threshold
        low_samples = samples < self.threshold
        high_samples = samples >= self.threshold

        # Compute the rising edge and falling edges by comparing the current value to the next with
        # the boolean operator & (if both are true the result is true) and converting this to an index
        # into the current array
        rising_edge_idx = np.nonzero(low_samples[:-1] & np.roll(high_samples, -1)[:-1])[0]
        falling_edge_idx = np.nonzero(high_samples[:-1] & np.roll(low_samples, -1)[:-1])[0]

        # This would need to be handled more gracefully with a stateful
        # processing (e.g. saving samples at the end if the pulse is in-between two processing blocks)
        # Remove stray falling edge at the start
        if len(rising_edge_idx) == 0 or len(falling_edge_idx) == 0:
            return
        #print(f"passed len test for idx's")
        if rising_edge_idx[0] > falling_edge_idx[0]:
            falling_edge_idx = falling_edge_idx[1:]

        # Remove stray rising edge at the end
        if rising_edge_idx[-1] > falling_edge_idx[-1]:
            rising_edge_idx = rising_edge_idx[:-1]

        rising_edge_diff = np.diff(rising_edge_idx)
        time_between_rising_edge = sample_rate / rising_edge_diff * 60

        pulse_widths = falling_edge_idx - rising_edge_idx
        rssi_idxs = list(np.arange(r, r + p) for r, p in zip(rising_edge_idx, pulse_widths))
        rssi = [np.mean(samples[r]) * max_samp for r in rssi_idxs]

        for t, r in zip(time_between_rising_edge, rssi):
            print(f"BPM: {t:.02f}")
            print(f"rssi: {r:.02f}")
        self.stateful_index += len(samples)
        print(f"stateful index : {self.stateful_index}")


class ReaderThread(threading.Thread):
    """Continuously read samples on a separate thread and place them on the buffer
    """
    reader: SampleReader
    buffer: SampleBuffer
    write_timeout: float = 1
    def __init__(self, reader: SampleReader, buffer: SampleBuffer) -> None:
        super().__init__()
        self.reader = reader
        self.buffer = buffer
        self.running: bool = False
        self._stopped = threading.Event()

    def run(self):
        self.running = True
        try:
            with self.reader:
                while self.running:
                    samples = self.reader.read_samples()
                    try:
                        self.buffer.put(samples, timeout=1)
                    except QueueFull:
                        print('Sample buffer overflow')
        finally:
            self._stopped.set()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._stopped.wait()



# NOTE: Always better to run things within a main function
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--read-only', dest='read_only', action='store_true')
    p.add_argument('-o', '--outfile', dest='outfile')
    p.add_argument('--max-samples', dest='max_samples', type=int)
    args = p.parse_args()
    if args.read_only:
        assert args.outfile is not None
        assert args.max_samples is not None
        run_readonly(args.outfile, args.max_samples)
    else:
        run_main(args.outfile, args.max_samples)

def run_readonly(outfile: str, max_samples: int):
    samples = np.zeros(0, dtype=np.complex128)
    reader = SampleReader()
    with reader:
        while samples.size < max_samples:
            _samples = reader.read_samples()
            samples = np.concatenate((samples, _samples))

    np.save(outfile, samples)

def run_main(outfile: str|None, max_samples: int|None):
    reader = SampleReader()
    processor = SampleProcessor(reader.sample_rate)
    buffer = SampleBuffer(maxsize=processor.num_samples_to_process * 3)

    samples = np.zeros(0, dtype=np.complex128)

    reader_thread = ReaderThread(reader, buffer)
    reader_thread.start()
    try:
        while True:
            _samples = processor.process_from_buffer(buffer)
            if outfile is not None:
                samples = np.concatenate((samples, _samples))
                if max_samples is not None and samples.size >= max_samples:
                    break
        if outfile is not None:
            np.save(outfile, samples)
    except KeyboardInterrupt:
        print('Closing...')
        reader_thread.stop()
        print('Closed')
        return


# NOTE: This only calls main() above ONLY when the script is being executed
#       This way you can import it without running the while loop
if __name__ == '__main__':
    main()
