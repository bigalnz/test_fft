import asyncio

import numpy as np
import pytest

from kiwitracker.common import ProcessConfig, SampleConfig
from kiwitracker.sample_processor import find_beep_frequencies
from kiwitracker.sample_reader import (chunk_numpy_file, pipeline,
                                       samples_source_file)


@pytest.mark.asyncio
async def test_scan(request, process_config):
    p = request.config.rootpath / "data" / "test_multiple_4.s8"

    process_config.carrier_freq = None
    process_config.sample_config.center_freq = 160_500_000
    process_config.sample_config.gain = 14.4

    samples_queue = asyncio.Queue()
    reader_task_fn = asyncio.create_task(
        samples_source_file(p, process_config.num_samples_to_process, None)(samples_queue)
    )

    find_beep_freqs_task = asyncio.create_task(find_beep_frequencies(samples_queue, process_config, N=13))

    result = await find_beep_freqs_task

    reader_task_fn.cancel()

    assert [160338981, 160708141] == result


@pytest.mark.asyncio
async def test_process_sample(request, async_queue_to_list, process_config):
    p = request.config.rootpath / "data" / "rtl_ct.s8"

    # asserts chunk_numpy_file() works correctly
    num_chunks = sum(1 for _ in chunk_numpy_file(p, np.dtype("int8"), 250_000))
    assert num_chunks == 1967

    data = []

    await pipeline(
        process_config=process_config,
        task_samples_input=samples_source_file(
            p,
            process_config.num_samples_to_process,
            num_chunks=50,
        ),
        task_results=async_queue_to_list(data),
    )

    bpms = [r.BPM for r in data]
    beep_durations = [r.BEEP_DURATION for r in data]

    bpms_expected = np.array(
        [
            100.75434568711051,
            79.20587856129947,
            76.72327672327671,
            80.13564627624885,
            74.08657904256602,
            69.89761092150171,
            78.66837387964148,
        ],
        dtype=np.float64,
    )

    beep_durations_expected = np.array(
        [
            0.01220703125,
            0.013780381944444444,
            0.012532552083333334,
            0.014105902777777778,
            0.013780381944444444,
            0.014865451388888888,
            0.012749565972222222,
        ],
        dtype=np.float64,
    )

    assert np.allclose(bpms, bpms_expected)
    assert np.allclose(beep_durations, beep_durations_expected)


@pytest.mark.asyncio
async def test_multiple_4_s8(request, async_queue_to_list):
    sc = SampleConfig(
        sample_rate=1_024_000,
        center_freq=160_500_000,
        gain=14.4,
    )

    pc = ProcessConfig(
        sample_config=sc,
        carrier_freq=160_708_082,
        running_mode="disk",
    )

    data = []
    await pipeline(
        process_config=pc,
        task_samples_input=samples_source_file(
            request.config.rootpath / "data" / "test_multiple_4.s8",
            pc.num_samples_to_process,
            num_chunks=None,
        ),
        task_results=async_queue_to_list(data),
    )

    bpms = [r.BPM for r in data]
    beep_durations = [r.BEEP_DURATION for r in data]

    bpms_expected = np.array(
        [
            153.29341317365268,
            79.9791720906014,
            79.9791720906014,
            79.9791720906014,
            80.07298318780137,
            79.9791720906014,
            79.95835502342528,
            80.0104180231801,
            80.05211726384364,
            79.93754879000781,
            79.94795055302538,
            80.07298318780137,
            79.98958468949355,
        ],
        dtype=np.float64,
    )
    beep_durations_expected = np.array(
        [
            0.012369791666666666,
            0.012152777777777776,
            0.011881510416666666,
            0.01171875,
            0.012586805555555554,
            0.012261284722222222,
            0.011881510416666666,
            0.012098524305555556,
            0.012586805555555554,
            0.011990017361111112,
            0.01144748263888889,
            0.012152777777777776,
            0.012152777777777776,
        ],
        dtype=np.float64,
    )

    assert np.allclose(bpms, bpms_expected)
    assert np.allclose(beep_durations, beep_durations_expected)
