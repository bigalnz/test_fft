import asyncio

import numpy as np
import pytest

from kiwitracker.common import ProcessConfig, ProcessResult, SampleConfig
from kiwitracker.sample_processor import (chick_timer, fast_telemetry,
                                          find_beep_frequencies)
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


def _ct_signal(d1, d2, signal_bpm=48.0):
    return [20.0, *[signal_bpm] * (d1 - 1), 16.0, *[signal_bpm] * (d2 - 1), 16.0]


@pytest.mark.asyncio
async def test_ft_synthetic(request, process_config):
    """
    Test FastTelemetry syntheticaly (without the samples file)
    """

    q_in = asyncio.Queue()
    q_out = asyncio.Queue()

    ft_task = asyncio.create_task(fast_telemetry(process_config, q_in, [q_out]))

    bpms_1 = [
        78.95,  # Mortality
        75.0,  # 3
        67.41,  # 9
        80.0,  # 0
        19.29,  # 7
    ]

    for b in bpms_1:
        r = ProcessResult(
            date=None, channel=59, BPM=b, DBFS=1, CLIPPING=1, BEEP_DURATION=1, SNR=1, latitude=0, longitude=0
        )
        q_in.put_nowait(r)

    await q_in.join()

    data = []
    while not q_out.empty():
        data.append(await q_out.get())
        q_out.task_done()

    ft_task.cancel()

    assert len(data) == 1
    assert data[0].mode == "Mortality"
    assert data[0].d1 == 39
    assert data[0].d2 == 7


@pytest.mark.asyncio
async def test_ct_synthetic(request, process_config):
    """
    Test ChickTimer syntheticaly (without the samples file)
    """

    q_in = asyncio.Queue()
    q_out = asyncio.Queue()

    ct_task = asyncio.create_task(chick_timer(process_config, q_in, [q_out]))

    bpms_1 = [
        *_ct_signal(2, 2),  # "days_since_change_of_state",
        *_ct_signal(2, 3),  # "days_since_hatch",
        *_ct_signal(2, 4),  # "days_since_desertion_alert",
        *_ct_signal(2, 5),  # "time_of_emergence",
        *_ct_signal(4, 6),  # "weeks_batt_life_left",
        *_ct_signal(2, 7),  # "activity_yesterday",
        *_ct_signal(2, 8),  # "activity_two_days_ago",
        *_ct_signal(2, 9),  # "mean_activity_last_four_days"
    ]

    for b in bpms_1:
        r = ProcessResult(
            date=None, channel=59, BPM=b, DBFS=1, CLIPPING=1, BEEP_DURATION=1, SNR=1, latitude=0, longitude=0
        )
        q_in.put_nowait(r)

    await q_in.join()

    data = []
    while not q_out.empty():
        data.append(await q_out.get())
        q_out.task_done()

    ct_task.cancel()

    assert len(data) == 1
    assert data[0].days_since_change_of_state == "22"
    assert data[0].days_since_hatch == "23"
    assert data[0].days_since_desertion_alert == "24"
    assert data[0].time_of_emergence == "25"
    assert data[0].weeks_batt_life_left == "46"
    assert data[0].activity_yesterday == "27"
    assert data[0].activity_two_days_ago == "28"
    assert data[0].mean_activity_last_four_days == "29"


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
            80.0,
            74.07764649143958,
            69.76268877029635,
            78.94128228189643,
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
            0.019694010416666664,
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
            80.0104180231801,
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
            0.019585503472222224,
        ],
        dtype=np.float64,
    )

    assert np.allclose(bpms, bpms_expected)
    assert np.allclose(beep_durations, beep_durations_expected)
