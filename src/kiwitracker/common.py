import datetime
import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from kiwitracker.gps import GPSBase, GPSDummy

SamplesT = npt.NDArray[np.complex128]
"""Alias for sample arrays"""

FloatArray = npt.NDArray[np.float64]

# frozen=True -> makes dataclass hashable


@dataclass
class SampleConfig:
    sample_rate: float = 1.024e6
    """Sample rate"""

    center_freq: float = 160_270_968
    """Center frequency"""

    read_size: int = 65536
    """Number of samples to read from the sdr in each iteration"""

    gain: str | float = 7.7
    """gain in dB"""

    bias_tee_enable: bool = False
    """Enable bias tee"""

    location: str = "Ponui"
    """Set population location"""


@dataclass
class ProcessConfig:
    sample_config: SampleConfig

    gps_module: GPSBase = field(default_factory=lambda: GPSDummy())

    # carrier_freq: float = 160_708_253
    carrier_freq: float = 160_270_968
    # carrier_freq: float = 160_274_340

    """Center frequency of the carrier wave to process (in Hz)"""

    num_samples_to_process: int = int(2.5e5)
    """Number of samples needed to process"""

    running_mode: str = "radio"
    """
        "radio" - read from radio
        "disk" - data are from disk
    """

    @property
    def sample_rate(self) -> float:
        return self.sample_config.sample_rate

    @property
    def freq_offset(self) -> float:
        return self.sample_config.center_freq - self.carrier_freq

    @property
    def channel(self) -> int:
        """Channel Number from Freq"""
        return math.floor((self.carrier_freq - 160.11e6) / 0.01e6)

    @property
    def fft_size(self, beep_duration=0.017) -> int:
        # this makes sure there's at least 1 full chunk within each beep
        return int(beep_duration * self.sample_rate / 2)

    scan: bool = False
    """Enable signal scanning"""


@dataclass
class ProcessResult:
    """
    One result from the sample processor
    """

    date: datetime.datetime
    channel: int
    BPM: float
    DBFS: float
    CLIPPING: float
    BEEP_DURATION: float
    SNR: float
    latitude: float
    longitude: float
