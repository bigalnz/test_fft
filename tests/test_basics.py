import numpy as np
import pytest

from kiwitracker.sample_reader import filename_to_dtype


def test_filename_to_dtype(tmp_npy_file_uint8, tmp_npy_file_uint32):
    assert filename_to_dtype("test.fc32") is np.dtype(np.complex64)
    assert filename_to_dtype("test.sc8") is np.dtype(np.uint8)
    assert filename_to_dtype("test.s8") is np.dtype(np.uint8)
    assert filename_to_dtype(tmp_npy_file_uint8) is np.dtype(np.uint8)
    assert filename_to_dtype(tmp_npy_file_uint32) is np.dtype(np.uint32)

    with pytest.raises(ValueError, match=r"Unknown file extension \.xxx\. .*"):
        filename_to_dtype("test.xxx")
