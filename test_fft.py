from typing import Self, TYPE_CHECKING
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



# Number of samples to read in each iteration
#samples_to_process = 4.8e6
threshold = 0.9
freq_offset = -43.8e4 #Hz

class SampleReader:
    sample_rate: float = 2.4e6
    center_freq: float = 160270968

    gain: float|str = 44.5
    """gain in dB"""

    num_samples: int = int(1.024e6)
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

    def read_samples(self) -> npt.NDArray[np.complex128]:
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


stateful_index = 0

def process_samples(samples, sample_rate, freq_offset, threshold):

    global stateful_index

    beep_duration = 0.017 # seconds
    fft_size = int(beep_duration * sample_rate / 2) # this makes sure there's at least 1 full chunk within each beep
    f = np.linspace(sample_rate/-2, sample_rate/2, fft_size)
    num_ffts = len(samples) // fft_size # // is an integer division which rounds down
    fft_thresh = 0.1
    beep_freqs = []
    for i in range(num_ffts):
        fft = np.abs(np.fft.fftshift(np.fft.fft(samples[i*fft_size:(i+1)*fft_size]))) / fft_size
        if np.max(fft) > fft_thresh:
            beep_freqs.append(np.linspace(sample_rate/-2, sample_rate/2, fft_size)[np.argmax(fft)])
        plt.plot(f,fft)
    #print(beep_freqs)
    #plt.show()

    t = np.arange(len(samples))/sample_rate
    samples = samples * np.exp(2j*np.pi*t*freq_offset)
    h = signal.firwin(501, 0.02, pass_zero=True)
    samples = np.convolve(samples, h, 'valid')
    samples = samples[::100]
    sample_rate = sample_rate/100
    samples = np.abs(samples)
    samples = np.convolve(samples, [1]*10, 'valid')/10
    max_samp = np.max(samples)
    # samples /= np.max(samples)
    #print(f"max sample : {max_samp}")
    #plt.plot(samples)
    #plt.show()

    # Get a boolean array for all samples higher or lower than the threshold
    low_samples = samples < threshold
    high_samples = samples >= threshold

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
    stateful_index += len(samples)
    print(f"stateful index : {stateful_index}")


# NOTE: Always better to run things within a main function
def main():
    with SampleReader() as reader:
        try:
            while True:
                # Read a chunk of samples
                # samples_to_process is number of samples to process
                #start = time.time()

                samples = reader.read_samples()
                #print(f"finish : {time.time()-start}")

                # use matplotlib to estimate and plot the PSD
                #psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
                #xlabel('Frequency (MHz)')
                #ylabel('Relative power (dB)')
                #show()

                # Process the samples

                processed_data = process_samples(
                    samples,
                    reader.sample_rate,
                    freq_offset,
                    threshold,
                )

        except KeyboardInterrupt:
            # Stop the loop on keyboard interrupt
            print("Program terminated.")
            return


# NOTE: This only calls main() above ONLY when the script is being executed
#       This way you can import it without running the while loop
if __name__ == '__main__':
    main()
