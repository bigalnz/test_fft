import asyncio
from pathlib import Path

import numpy as np
import pytest

from kiwitracker.common import ProcessConfig, SampleConfig
from kiwitracker.sample_reader import chunk_numpy_file, run_from_disk_2


@pytest.mark.asyncio
async def test_process_sample(request, process_config):
    p = Path(request.config.rootpath) / "data" / "rtl_ct.s8"

    # assert chunk_numpy_file() works correctly
    num_chunks = sum(1 for _ in chunk_numpy_file(p, np.dtype("uint8"), 250_000))
    assert num_chunks == 1967

    # assert process_sample() works correctly
    # we are reading first 50 chunks and process them
    # the result is stored in `out_queue`
    out_queue = asyncio.Queue()
    await run_from_disk_2(process_config, p, out_queue, num_chunks=50)

    data = []
    while not out_queue.empty():
        data.append(await out_queue.get())
        out_queue.task_done()

    bpms = [r.BPM for r in data]
    beep_durations = [r.BEEP_DURATION for r in data]

    bpms_expected = np.array(
        [186.01271571298818, 30.0, 17.316798196166854, 29.998535227772084, 30.32875900878665], dtype=np.float64
    )

    beep_durations_expected = np.array(
        [0.11015625, 0.11025390625, 0.0896484375, 0.08955078125, 0.0033203125], dtype=np.float64
    )

    assert np.allclose(bpms, bpms_expected)
    assert np.allclose(beep_durations, beep_durations_expected)


@pytest.mark.asyncio
async def test_multiple_4_s8(request):
    p = Path(request.config.rootpath) / "data" / "test_multiple_4.s8"

    sc = SampleConfig(sample_rate=1_024_000, center_freq=160_500_000, gain=14.4)
    pc = ProcessConfig(sample_config=sc, carrier_freq=160_708_082, running_mode="disk")

    # ProcessConfig(sample_config=SampleConfig(sample_rate=1024000.0, center_freq=160500000.0, read_size=65536, gain=14.4, bias_tee_enable=False, location='Ponui'), carrier_freq=160708082.0, num_samples_to_process=250000)

    out_queue = asyncio.Queue()
    await run_from_disk_2(pc, p, out_queue)

    data = []
    while not out_queue.empty():
        data.append(await out_queue.get())
        out_queue.task_done()

    bpms = [r.BPM for r in data]
    beep_durations = [r.BEEP_DURATION for r in data]

    bpms_expected = np.array([0.13997102162442931, 13.539600687557847, 1.491215083019225], dtype=np.float64)
    beep_durations_expected = np.array([0.00166015625, 0.00234375, 0.00048828125], dtype=np.float64)

    print(bpms)
    print(beep_durations)

    # assert np.allclose(bpms, bpms_expected)
    # assert np.allclose(beep_durations, beep_durations_expected)
