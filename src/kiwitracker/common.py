import datetime
import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from kiwitracker.gps import GPSBase, GPSDummy

SamplesT = npt.NDArray[np.complex128]
"""Alias for sample arrays"""

FloatArray = npt.NDArray[np.float64]


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

    scan_interval: int | None = None
    """Scan interval in minutes"""


@dataclass
class ProcessConfig:
    sample_config: SampleConfig

    gps_module: GPSBase = field(default_factory=lambda: GPSDummy())

    # carrier_freq: float = 160_708_253
    carrier_freq: float = 160_270_968
    # carrier_freq: float = 160_274_340

    """Center frequency of the carrier wave to process (in Hz)"""

    num_samples_to_process: int = 256_000  # 256k int(2.5e5)
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
        return int(round((self.carrier_freq - 160.12e6) / 0.01e6))

    @property
    def fft_size(self, beep_duration=0.017) -> int:
        # this makes sure there's at least 1 full chunk within each beep
        return int(beep_duration * self.sample_rate / 2)


@dataclass
class ProcessResult:
    """
    One result from process_sample()
    """

    date: datetime.datetime
    channel: int
    carrier_freq: float
    BPM: float
    DBFS: float
    CLIPPING: float
    BEEP_DURATION: float
    SNR: float
    latitude: float
    longitude: float


@dataclass
class CTResult:
    """
    One result from chick_timer()
    """

    channel: int
    carrier_freq: float
    decoding_success: bool

    start_dt: datetime.datetime
    end_dt: datetime.datetime

    snr_min: float
    snr_max: float
    snr_mean: float

    dbfs_min: float
    dbfs_max: float
    dbfs_mean: float

    lat: float
    lon: float

    days_since_change_of_state: int | None = None
    days_since_hatch: int | None = None
    days_since_desertion_alert: int | None = None
    time_of_emergence: int | None = None
    weeks_batt_life_left: int | None = None
    activity_yesterday: int | None = None
    activity_two_days_ago: int | None = None
    mean_activity_last_four_days: int | None = None


@dataclass
class FTResult:
    """
    One result from fast_telemetry()
    """

    channel: int
    carrier_freq: float

    start_dt: datetime.datetime
    end_dt: datetime.datetime

    snr_min: float
    snr_max: float
    snr_mean: float

    dbfs_min: float
    dbfs_max: float
    dbfs_mean: float

    lat: float
    lon: float

    mode: str
    d1: int
    d2: int
