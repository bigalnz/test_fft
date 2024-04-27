import asyncio

import numpy as np
import pytest

from kiwitracker.common import ProcessConfig, SampleConfig
from kiwitracker.sample_processor import find_beep_frequencies
from kiwitracker.sample_reader import chunk_numpy_file, pipeline, source_file


@pytest.mark.asyncio
async def test_scan(request, process_config):
    p = request.config.rootpath / "data" / "test_multiple_4.s8"

    process_config.carrier_freq = None
    process_config.sample_config.center_freq = 160_500_000
    process_config.sample_config.gain = 14.4

    find_beep_freqs_task = asyncio.create_task(
        find_beep_frequencies(
            source_file(
                p,
                process_config.num_samples_to_process,
                None,
            ),
            process_config,
            N=13,
        )
    )

    result = await find_beep_freqs_task

    assert [160338981, 160708043] == result


@pytest.mark.asyncio
async def test_process_sample(request, async_queue_to_list, process_config):
    p = request.config.rootpath / "data" / "rtl_ct.s8"

    # asserts chunk_numpy_file() works correctly
    num_chunks = sum(1 for _ in chunk_numpy_file(p, np.dtype("int8"), 250_000))
    assert num_chunks == 1967

    data = []

    await pipeline(
        process_config=process_config,
        source_gen=source_file(
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
            101.95818121473614, 
            78.94128228189643,
            76.92500313008638,
            80.0, 74.07764649143958, 
            69.76268877029635, 
            78.94128228189643
         ],
        dtype=np.float64,
    )

    beep_durations_expected = np.array(
        [
            0.019694010416666664,
            0.019639756944444444, 
            0.019694010416666664, 
            0.019694010416666664, 
            0.019748263888888888, 
            0.019694010416666664, 
            0.019694010416666664
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
        source_gen=source_file(
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
            155.8599695585997,
            80.0,
            80.0,
            79.98958468949355,
            80.0, 
            80.0104180231801,
            79.9791720906014,
            79.98958468949355,
            80.02083876009377,
            79.98958468949355,
            79.98958468949355,
            80.0104180231801,
            80.0104180231801
        ],
        dtype=np.float64,
    )
    beep_durations_expected = np.array(
        [
            0.01953125, 
            0.01953125,
            0.019585503472222224,
            0.019476996527777776,
            0.01953125, 
            0.019585503472222224, 
            0.019422743055555552, 
            0.019368489583333336, 
            0.019476996527777776, 
            0.019422743055555552, 
            0.019314236111111112, 
            0.01953125, 
            0.019585503472222224
        ],
        dtype=np.float64,
    )

    print(bpms)
    print(beep_durations)

    assert np.allclose(bpms,bpms_expected)
    assert np.allclose(beep_durations, beep_durations_expected)
