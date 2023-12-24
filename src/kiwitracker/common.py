from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

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

    gain: str|float = 7.7
    """gain in dB"""


@dataclass
class ProcessConfig:
    sample_config: SampleConfig

    carrier_freq: float = 160_707_760
    """Center frequency of the carrier wave to process (in Hz)"""

    num_samples_to_process: int = int(2.56e5)
    """Number of samples needed to process"""
