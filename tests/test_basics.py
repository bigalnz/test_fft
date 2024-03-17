import numpy as np
import pytest

from kiwitracker.sample_reader import filename_to_dtype


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


def test_filename_to_dtype(tmp_npy_file_uint8, tmp_npy_file_uint32):
    assert filename_to_dtype("test.fc32") is np.complex64
    assert filename_to_dtype("test.sc8") is np.uint8
    assert filename_to_dtype("test.s8") is np.uint8
    assert filename_to_dtype(tmp_npy_file_uint8) is np.uint8
    assert filename_to_dtype(tmp_npy_file_uint32) is np.uint32

    with pytest.raises(ValueError, match=r"Unknown file extension \.xxx\. .*"):
        filename_to_dtype("test.xxx")
