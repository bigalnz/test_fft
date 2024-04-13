"""
conftest.py - pytest will automatically detect this file
and use fixtures defined here in all tests
"""

import logging

import numpy as np
import pytest

from kiwitracker.common import ProcessConfig, SampleConfig


@pytest.fixture(scope="session")
def tmp_npy_file_uint8(tmp_path_factory):
    arr = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    fn = tmp_path_factory.mktemp("data") / "test_uint8.npy"
    np.save(fn, arr)
    return fn


@pytest.fixture(scope="session")
def tmp_npy_file_uint32(tmp_path_factory):
    arr = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    fn = tmp_path_factory.mktemp("data") / "test_uint32.npy"
    np.save(fn, arr)
    return fn


@pytest.fixture(scope="function")
def sample_config():
    return SampleConfig(
        sample_rate=1024000.0,
        center_freq=160270968.0,
        read_size=65536,
        gain=33.8,
        bias_tee_enable=False,
        location="Ponui",
    )


@pytest.fixture(scope="function")
def process_config(sample_config):
    return ProcessConfig(
        sample_config=sample_config,
        carrier_freq=160707530.0,
        num_samples_to_process=250000,
        running_mode="disk",
    )


@pytest.fixture(scope="session")
def logger():
    l = logging.getLogger("KiwiTracker")
    l.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    l.addHandler(ch)
    return l
