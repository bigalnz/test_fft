import ctypes
import time
from logging import getLogger

import numpy

"""

Python Wrapper to use airspyhf user mode driver
https://github.com/airspy/airspyhf

"""

logger = getLogger("KiwiTracker")


class VER(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("rev", ctypes.c_uint32),
    ]


class AirspyHfTransfer(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_void_p),
        ("ctx", ctypes.c_void_p),
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("samples_count", ctypes.c_int),
        ("dropped_samples", ctypes.c_uint64),
    ]


clibrary = ctypes.CDLL("libairspyhf.so")  # setup instance of the ctypes ref

# airspyhf_lib_version
clibrary.airspyhf_lib_version.restype = None  # define the return type of the c function using python type 'None'
clibrary.airspyhf_lib_version.argtypes = [
    ctypes.POINTER(VER)
]  # declare the method types for the params using the struct

# airspyhf_lib_get_serial
clibrary.airspyhf_list_devices.restype = ctypes.c_int
clibrary.airspyhf_list_devices.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_int]

# airspyhf_lib_get_device
clibrary.airspyhf_open_sn.restype = ctypes.c_int
clibrary.airspyhf_open_sn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64]

# airspyhf_get_samp_rates
clibrary.airspyhf_get_samplerates.restype = ctypes.c_int
clibrary.airspyhf_get_samplerates.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]

# airspyhf_set_samp_rate
clibrary.airspyhf_set_samplerate.restype = ctypes.c_int
clibrary.airspyhf_set_samplerate.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

# airspyhf_set_freq
clibrary.airspyhf_set_freq.restype = ctypes.c_int
clibrary.airspyhf_set_freq.argtypes = [ctypes.c_void_p, ctypes.c_uint32]


def get_version():
    # GET VERSION NUMBERS
    ver = VER()  # new instance of struct
    clibrary.airspyhf_lib_version(ctypes.byref(ver))  # pass it in, using byref as the instance of ver will be filled
    # print(f" Ver : {ver.major}.{ver.minor}.{ver.rev}")  # print the result. No idea why contents didnt work
    return ver


def get_serial():
    # GET SERIAL NUMBER
    serial = numpy.zeros(1, dtype="uint64")
    devices = clibrary.airspyhf_list_devices(serial.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), 1)
    # print(f" Number of devices is {devices}")
    # print(hex(serial[0]))  # 0xdc52978027444a4d
    return serial[0]


def open_device(serial):
    # GET POINTER TO DEVICE
    device_handle = ctypes.c_void_p(0)
    sn = ctypes.c_uint64(serial)
    status = clibrary.airspyhf_open_sn(ctypes.byref(device_handle), sn)
    # print(status)
    return device_handle


def get_sample_rates(device_handle):
    # GET NUM SAMP RATES
    how_many_sample_rates = ctypes.c_uint32(0)
    status = clibrary.airspyhf_get_samplerates(device_handle, ctypes.byref(how_many_sample_rates), ctypes.c_uint32(0))
    # print("Status =", status, "how_many_sample_rates =", how_many_sample_rates)
    num_rates = how_many_sample_rates.value
    # GET SAMP RATES
    rates = numpy.zeros(num_rates, dtype="uint32")
    status = clibrary.airspyhf_get_samplerates(
        device_handle, rates.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), ctypes.c_uint32(num_rates)
    )
    # print("Status =", status, "rates =", rates)
    return rates


def set_freq(device_handle, freq):
    # device, const uint32 freq_hz
    status = clibrary.airspyhf_set_freq(device_handle, ctypes.c_uint32(int(freq)))
    # print("Set Freq Status =", status)
    return status


def set_sample_rate(device_handle, rate):
    status = clibrary.airspyhf_set_samplerate(device_handle, ctypes.c_uint32(int(rate)))
    # print("Set Rate Status =", status)
    return status


def set_default_options(device_handle):
    clibrary.airspyhf_set_hf_lna(device_handle, ctypes.c_uint8(1))  # LNA on
    clibrary.airspyhf_set_hf_agc(device_handle, ctypes.c_uint8(0))  # AGC off

    clibrary.airspyhf_set_att(device_handle, ctypes.c_float(0.0))  # ATT 0
    clibrary.airspyhf_set_hf_att(device_handle, ctypes.c_uint8(0))  # ATT 0


@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(AirspyHfTransfer))
def _rx_callback(transfer):
    if transfer.contents.dropped_samples > 0:
        logger.error(f"Dropped samples {transfer.contents.dropped_samples}")

    # complex_data = numpy.ctypeslib.as_array(transfer.contents.samples, shape=(transfer.contents.samples_count, 2)).view(
    #     "complex64"
    # )
    complex_data = (
        numpy.asarray(transfer.contents.samples.contents, dtype="float32")
        .reshape(transfer.contents.samples_count, 2)
        .view("complex64")
    )

    callback_fn = ctypes.cast(transfer.contents.ctx, ctypes.POINTER(ctypes.py_object)).contents.value
    callback_fn(complex_data)

    return 0


_pointers_to_callbacks = []  # prevent garbage-collect ctypes.pointer to callback_fns


def start_sampling(device_handle, callback_fn):
    ppyo = ctypes.pointer(ctypes.py_object(callback_fn))
    _pointers_to_callbacks.append(ppyo)

    clibrary.airspyhf_start(device_handle, _rx_callback, ctypes.cast(ppyo, ctypes.c_void_p))


if __name__ == "__main__":
    sn = get_serial()
    device_handle = open_device(sn)
    rates = get_sample_rates(device_handle)
    print(rates[0])
    set_freq(device_handle, 160500000)
    set_sample_rate(device_handle, rates[0])

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(AirspyHfTransfer))
    def rx_callback(transfer):
        print("SAMPLE COUNT =", transfer.contents.samples_count)
        print("SAMPLES = ", transfer.contents.samples)
        complex_data = numpy.ctypeslib.as_array(
            transfer.contents.samples, shape=(transfer.contents.samples_count, 2)
        ).view("complex64")
        print(complex_data)
        return 0

    def airspyhf_start(device_handle, callback_fn, ctx=None):
        return clibrary.airspyhf_start(device_handle, callback_fn, ctx)

    airspyhf_start(device_handle, rx_callback)

    while True:
        time.sleep(1)
