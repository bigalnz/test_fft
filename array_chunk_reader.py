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



def main():
    # processor = SampleProcessor(2400000)
    samples = np.load('samples.npy')
    for i in range(0, samples.size[0], 240000):
        sample_to_process = samples[i*samples.size, (i*samples.size)+i ]
        # processor.process(samples)
main()
