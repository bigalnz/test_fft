import asyncio
from pathlib import Path

import numpy as np
import pytest

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
