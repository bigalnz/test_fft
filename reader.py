from typing import Self, TYPE_CHECKING
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

class SampleReader:

    sample_rate: float = 2.4e6
    center_freq: float = 160270968
    gain: float|str = 44.5
    sdr: RtlSdr|None = None

    num_samples_to_process: int = int(2.4e6)
    
    def open(self):
        sdr = self.sdr = RtlSdr()
        sdr.sample_rate = self.sample_rate
        sdr.center_freq = self.center_freq
        sdr.gain = self.gain

    def read_samples(self):
        samples = self.sdr.read_samples(self.num_samples_to_process)
        # print(f" samp size : {samples.size}")
        return samples
    
    def close(self):
        sdr = self.sdr
        self.sdr = None
        sdr.close()

class SampleProcessor:
    threshold: float = 0.8
    freq_offset: float = -43.8e4 #Hz
    beep_duration: float = 0.017 # seconds
    num_samples_to_process: int = int(2.4e5)

    sample_rate: float
    stateful_index: int
    stateful_rising_edge: int

    def __init__(self, sample_rate: float) -> None:
        self.sample_rate = sample_rate
        self.stateful_index = 0
        self.stateful_rising_edge = 0

    def process(self, samples):
        # do something here with samples that needs to be passed
        t = np.arange(len(samples))/self.sample_rate
        samples = samples * np.exp(2j*np.pi*t*self.freq_offset)
        h = signal.firwin(501, 0.02, pass_zero=True)
        samples = np.convolve(samples, h, 'valid')
        samples = samples[::100]
        sample_rate = self.sample_rate/100
        samples = np.abs(samples)
        samples = np.convolve(samples, [1]*10, 'valid')/10
        max_samp = np.max(samples)

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
            self.stateful_index += samples.size + 14
            return
        #print(f"passed len test for idx's")
        if rising_edge_idx[0] > falling_edge_idx[0]:
            falling_edge_idx = falling_edge_idx[1:]

        # Remove stray rising edge at the end
        if rising_edge_idx[-1] > falling_edge_idx[-1]:
            rising_edge_idx = rising_edge_idx[:-1]

        samples_between =  (rising_edge_idx[0]+self.stateful_index) - self.stateful_rising_edge
        time_between = 1/sample_rate * samples_between
        pulse_per_minute = 60 / time_between
        self.stateful_rising_edge = self.stateful_index + rising_edge_idx[0]
        print(f" ppm : {pulse_per_minute}")

        # increment sample count    
        self.stateful_index += samples.size + 14
        
def main():
    
    
    samples_buffer: SamplesT = np.zeros(0, dtype=np.complex128)
    reader = SampleReader()
    processor = SampleProcessor(reader.sample_rate)
    reader.open()

    try:
          while True:
            # processor.process_from_buffer(buffer)
            # print("inside true")
            time_start = time.time()
            samples = reader.read_samples()
            # if samples not amount to 10s worth then keep adding
            if samples_buffer.size < 24e6:
                samples_buffer = np.concatenate((samples_buffer, samples))
                print(f"total time : {time.time() - time_start}")
                processor.process(samples)

            # if samples at 24M then close sdr and write to file
            if samples_buffer.size > 24e6:
                # close reading
                reader.close()
                # write buffer to file
                with open('samples_parallel.npy', 'wb') as f:
                    np.save(f, samples_buffer)
                return

    except KeyboardInterrupt:
        print('Closing...')
        reader_thread.stop()
        print('Closed')
        return
main()

    
