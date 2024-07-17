from __future__ import annotations

import argparse
import asyncio
# import cProfile
import io
import logging
import os
import pstats
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from types import FunctionType
from typing import AsyncIterator, Callable

import numpy as np
from sqlalchemy.orm import Session

from kiwitracker.common import ProcessConfig, ProcessResult, SampleConfig
from kiwitracker.db.engine import (construct_db_connection_string,
                                   construct_sqlalchemy_engine,
                                   get_sqlalchemy_engine, migrate_if_needed)
from kiwitracker.db.models import BPM, ChickTimerResult, FastTelemetryResult
from kiwitracker.exceptions import CarrierFrequencyNotFound
from kiwitracker.gps import GPSDummy, GPSReal
from kiwitracker.logging import setup_logging
from kiwitracker.sample_processor import (chick_timer, fast_telemetry,
                                          find_beep_frequencies,
                                          process_sample, process_sample_new)

# tracemalloc.start()

logger = logging.getLogger("KiwiTracker")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-f",
        "--from-file",
        dest="infile",
        help="Read samples from the given filename and process them",
    )
    p.add_argument(
        "-db",
        "--database",
        dest="db",
        default=None,
        help="SQLite database where to store processed results. Defaults to `main.db`. Environment variable KIWITRACKER_DB has priority.",
    )

    p.add_argument(
        "-d",
        "--delete-database",
        dest="deletedb",
        action="store_true",
        help="If SQLite database file exists upon start, it is deleted.",
    )

    p.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        help="Read samples from the device and save to the given filename",
    )
    p.add_argument(
        "-m",
        "--max-samples",
        dest="max_samples",
        type=int,
        help='Number of samples to read when "-o/--outfile" is specified',
    )
    p.add_argument(
        "--scan",
        dest="scan",
        type=int,
        nargs="?",
        const=0,
        help="Scan periodically for frequencies 0-240 (in minutes, default: %(default)s). 0 means scan only upon startup, cannot be used with --carrier flag",
    )
    p.add_argument(
        "--no-use-gps",
        dest="no_use_gps",
        action="store_true",
        help="Set this flag to not use GPS module",
    )
    p.add_argument(
        "--radio",
        default="rtl",
        const="rtl",
        nargs="?",
        choices=["rtl", "airspy", "dummy"],
        help="type of radio to be used (default: %(default)s), ignored if reading samples from disk.",
    )

    s_group = p.add_argument_group("Sampling")
    s_group.add_argument(
        "-c",
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=SampleConfig.read_size,
        help="Chunk size for sdr.read_samples (default: %(default)s)",
    )
    s_group.add_argument(
        "-s",
        "--sample-rate",
        dest="sample_rate",
        type=float,
        default=SampleConfig.sample_rate,
        help="SDR sample rate (default: %(default)s)",
    )
    s_group.add_argument(
        "--center-freq",
        dest="center_freq",
        type=float,
        default=SampleConfig.center_freq,
        help="SDR center frequency (default: %(default)s)",
    )
    s_group.add_argument(
        "-g",
        "--gain",
        dest="gain",
        type=float,
        default=SampleConfig.gain,
        help="SDR gain (default: %(default)s)",
    )
    s_group.add_argument(
        "--bias-tee",
        dest="bias_tee",
        action="store_true",
        help="Enable bias tee",
    )

    s_group.add_argument(
        "-log", "--loglevel", default="warning", help="Provide logging level. Example --loglevel debug, default=warning"
    )

    p_group = p.add_argument_group("Processing")
    p_group.add_argument(
        "--carrier",
        dest="carrier",
        type=float,
        nargs="?",
        const=ProcessConfig.carrier_freq,
        # default=ProcessConfig.carrier_freq,
        help="Carrier frequency to process (default: %(default)s)",
    )

    args = p.parse_args()

    setup_logging(level=args.loglevel.upper())

    if args.deletedb:
        db_filename = construct_db_connection_string(db_file=args.db).removeprefix("sqlite:///")

        if os.path.exists(db_filename):
            logger.info(f"Deleting DB file {db_filename}")
            os.remove(db_filename)

    construct_sqlalchemy_engine(db_file=args.db)
    engine = get_sqlalchemy_engine()

    logger.info(f"Using DB connection URL: {engine.url.render_as_string(hide_password=False)}")

    migrate_if_needed(engine, "head")

    if args.scan is not None and args.carrier is not None:
        logger.error("--scan and --carrier cannot be used simultaneously.")
        return
    elif args.scan is None and args.carrier is None:
        args.carrier_freq = ProcessConfig.carrier_freq

    if args.scan is not None and not 0 <= args.scan <= 240:
        logger.error(f"Scan interval outside valid range 0-240: ({args.scan})")
        return

    if args.no_use_gps:
        gps_module = GPSDummy()
    else:
        gps_module = GPSReal()

    gps_module.connect()

    sample_config = SampleConfig(
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        gain=args.gain,
        bias_tee_enable=args.bias_tee,
        read_size=args.chunk_size,
        scan_interval=args.scan,
    )

    process_config = ProcessConfig(
        sample_config=sample_config,
        carrier_freq=args.carrier,
        gps_module=gps_module,
    )

    if args.infile is not None:
        # import cProfile
        # import io
        # import pstats
        # from pstats import SortKey

        # pr = cProfile.Profile()

        # pr.enable()

        process_config.running_mode = "disk"

        asyncio.run(
            pipeline(
                process_config=process_config,
                source_gen=source_file(
                    filename=args.infile,
                    N=process_config.num_samples_to_process,
                    num_chunks=100,
                ),
                task_results=results_pipeline,
            )
        )

        # pr.disable()
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats()
        # print(s.getvalue())

    elif args.outfile is not None:
        logger.error("Readonly mode not implemented (yet.)")
        return
    else:
        if args.radio == "rtl":
            from kiwitracker.sample_buffer import SampleBuffer
            from kiwitracker.sample_reader_rtlsdr import \
                SampleReaderRtlSdr as SampleReader

            process_config.running_mode = "radio"
            asyncio.run(
                pipeline(
                    process_config=process_config,
                    source_gen=source_radio(
                        reader=SampleReader(sample_config),
                        buffer=SampleBuffer(maxsize=process_config.num_samples_to_process * 3),
                        num_samples_to_process=process_config.num_samples_to_process,
                    ),
                    task_results=results_pipeline,
                )
            )

        elif args.radio == "airspy":
            from kiwitracker.sample_reader_airspy import \
                BufferAirspy as SampleBuffer
            from kiwitracker.sample_reader_airspy import \
                SampleReaderAirspy as SampleReader

            process_config.running_mode = "radio"
            asyncio.run(
                pipeline(
                    process_config=process_config,
                    source_gen=source_radio(
                        reader=SampleReader(sample_config),
                        buffer=SampleBuffer(),
                        num_samples_to_process=process_config.num_samples_to_process,
                    ),
                    task_results=results_pipeline,
                )
            )

        elif args.radio == "dummy":

            from kiwitracker.sample_reader_dummy import \
                BufferDummy as SampleBuffer
            from kiwitracker.sample_reader_dummy import \
                SampleReaderDummy as SampleReader

            process_config.running_mode = "radio"
            asyncio.run(
                pipeline(
                    process_config=process_config,
                    source_gen=source_radio(
                        reader=SampleReader(sample_config),
                        buffer=SampleBuffer(),
                        num_samples_to_process=process_config.num_samples_to_process,
                    ),
                    task_results=results_pipeline,
                )
            )

        else:
            raise ValueError(f"Unknown radio value: {args.radio}")

    # asyncio.run(run_main_2(sample_config=sample_config, process_config=process_config))

    # if args.infile is not None:
    #     process_config.running_mode = "disk"

    #     run_from_disk(
    #         process_config=process_config,
    #         filename=args.infile,
    #     )
    # elif args.outfile is not None:
    #     assert args.max_samples is not None
    #     asyncio.run(
    #         run_readonly(
    #             sample_config=sample_config,
    #             filename=args.outfile,
    #             max_samples=args.max_samples,
    #         )
    #     )
    # else:
    #     asyncio.run(run_main(sample_config=sample_config, process_config=process_config))


async def store_bpm_to_db(queue: asyncio.Queue) -> None:
    with Session(get_sqlalchemy_engine()) as db_session:
        while True:
            r = await queue.get()

            bpm = BPM(
                dt=r.date,
                channel=r.channel,
                bpm=r.BPM,
                dbfs=r.DBFS,
                clipping=r.CLIPPING,
                duration=r.BEEP_DURATION,
                snr=r.SNR,
                lat=r.latitude,
                lon=r.longitude,
            )

            db_session.add(bpm)

            db_session.commit()
            queue.task_done()


async def store_ft_to_db(queue: asyncio.Queue) -> None:
    with Session(get_sqlalchemy_engine()) as db_session:
        while True:
            r = await queue.get()

            ft = FastTelemetryResult(
                channel=r.channel,
                carrier_freq=r.carrier_freq,
                start_dt=r.start_dt,
                end_dt=r.end_dt,
                snr_min=r.snr_min,
                snr_max=r.snr_max,
                snr_mean=r.snr_mean,
                dbfs_min=r.dbfs_min,
                dbfs_max=r.dbfs_max,
                dbfs_mean=r.dbfs_mean,
                lat=r.lat,
                lon=r.lon,
                mode=r.mode,
                d1=r.d1,
                d2=r.d2,
            )

            db_session.add(ft)
            db_session.commit()

            queue.task_done()


async def store_ct_to_db(queue: asyncio.Queue) -> None:
    with Session(get_sqlalchemy_engine()) as db_session:
        while True:
            r = await queue.get()

            ct = ChickTimerResult(
                channel=r.channel,
                carrier_freq=r.carrier_freq,
                decoding_success=r.decoding_success,
                start_dt=r.start_dt,
                end_dt=r.end_dt,
                snr_min=r.snr_min,
                snr_max=r.snr_max,
                snr_mean=r.snr_mean,
                dbfs_min=r.dbfs_min,
                dbfs_max=r.dbfs_max,
                dbfs_mean=r.dbfs_mean,
                lat=r.lat,
                lon=r.lon,
                days_since_change_of_state=r.days_since_change_of_state,
                days_since_hatch=r.days_since_hatch,
                days_since_desertion_alert=r.days_since_desertion_alert,
                time_of_emergence=r.time_of_emergence,
                weeks_batt_life_left=r.weeks_batt_life_left,
                activity_yesterday=r.activity_yesterday,
                activity_two_days_ago=r.activity_two_days_ago,
                mean_activity_last_four_days=r.mean_activity_last_four_days,
            )

            db_session.add(ct)
            db_session.commit()

            queue.task_done()


async def results_pipeline(
    pc: ProcessConfig,
    queue: asyncio.Queue,
) -> None:

    # while True:
    #     _ = await queue.get()
    #     queue.task_done()

    # return

    fast_telemetry_queue = asyncio.Queue()
    chick_timer_queue = asyncio.Queue()
    store_bpm_to_db_queue = asyncio.Queue()
    store_ct_to_db_queue = asyncio.Queue()
    store_ft_to_db_queue = asyncio.Queue()

    tasks = [
        asyncio.create_task(store_bpm_to_db(store_bpm_to_db_queue)),
        asyncio.create_task(chick_timer(pc, chick_timer_queue, [store_ct_to_db_queue])),
        asyncio.create_task(fast_telemetry(pc, fast_telemetry_queue, [store_ft_to_db_queue])),
        asyncio.create_task(store_ct_to_db(store_ct_to_db_queue)),
        asyncio.create_task(store_ft_to_db(store_ft_to_db_queue)),
    ]

    while True:
        bpm_result = await queue.get()

        for q in (store_bpm_to_db_queue, chick_timer_queue, fast_telemetry_queue):
            await q.put(bpm_result)

        queue.task_done()

    # wait for queues
    for q in (
        store_bpm_to_db_queue,
        chick_timer_queue,
        store_ct_to_db_queue,
        store_ft_to_db_queue,
        fast_telemetry_queue,
    ):
        await q.join()

    # cancel all tasks
    for t in tasks:
        t.cancel()


# async def run_readonly(sample_config: SampleConfig, filename: str, max_samples: int):
#     chunk_size = sample_config.read_size
#     nrows = max_samples // sample_config.read_size
#     if nrows * chunk_size < max_samples:
#         nrows += 1
#     samples = np.zeros((nrows, chunk_size), dtype=np.complex128)
#     sample_config = SampleConfig(read_size=chunk_size)
#     reader = SampleReader(sample_config)

#     async with reader:
#         await reader.open_stream()
#         i = 0
#         count = 0
#         async for _samples in reader:
#             if count == 0:
#                 print(f"{_samples.size=}")
#             samples[i, :] = _samples
#             count += _samples.size
#             # print(f'{i}\t{reader.aio_queue.qsize()=}\t{count=}')
#             i += 1
#             if count >= max_samples:
#                 break
#     samples = samples.flatten()[:max_samples]
#     np.save(filename, samples)


def filename_to_dtype(filename):
    file_extension = os.path.splitext(filename)[1]
    file_dtype = np.complex64

    match file_extension:
        case ".fc32":
            file_dtype = np.dtype(np.complex64)
        case ".sc8":
            file_dtype = np.dtype(np.int8)
        case ".s8":
            file_dtype = np.dtype(np.uint8)
        case ".npy":
            # read the sample data type from the first sample in the file
            file_dtype = np.dtype(type(np.load(filename, mmap_mode="r")[0]))
        case _:
            raise ValueError(f"Unknown file extension {file_extension}. Rename to one of .fc32, .sc8, .s8 or .npy")

    return file_dtype


def chunk_numpy_file(filename, dtype, N):
    current_offset = 0
    while True:
        arr = np.fromfile(filename, dtype=dtype, count=N, offset=current_offset)

        # Convert unsigned 8 bit samples to 32 bit floats and complex
        # https://k3xec.com/packrat-processing-iq/ (RTL-SDR part)
        if dtype == "uint8":
            iq = arr.astype(np.float32).view(np.complex64)  # 255 + 255j   0 + 0j
            iq /= 127.5  # 2 + 2j       0 + 0j
            iq -= 1 + 1j  # 1 + 1j      -1 - 1j
            arr = iq.copy()
        elif dtype == "int8":
            iq = arr.astype(np.float32).view(np.complex64)  # 128 + 128j  -127 - 127j
            iq /= 128  # 1 + 1j      -0.992 - 0.992j
            arr = iq.copy()
        elif dtype == "complex128":
            arr = arr.astype(np.complex64)

        if len(arr) == 0:
            break

        yield arr

        current_offset += N * dtype.itemsize


async def source_file(
    filename: str,
    N: int,
    num_chunks: int | None,
) -> AsyncIterator[np.ndarray]:
    """
    filename          -> file to read from
    N                 -> from ProcessConfig.num_samples_to_process
    num_chunks        -> how many chunks to read (or None for all)
    """

    file_dtype = filename_to_dtype(filename)

    for chunks_processed, chunk in enumerate(chunk_numpy_file(filename, file_dtype, N), 1):
        yield chunk
        await asyncio.sleep(0)

        if num_chunks is not None and chunks_processed >= num_chunks:
            break


async def source_radio(
    reader,
    buffer,
    num_samples_to_process: int,
) -> AsyncIterator[np.ndarray]:
    """
    reader                  -> the radio
    buffer                  -> where to put samples from the radio
    num_samples_to_process  -> number of samples in one chunk
    """

    if isinstance(reader, FunctionType):
        reader = reader()

    reader.buffer = buffer
    async with reader:
        await reader.open_stream()
        while True:
            chunk = await buffer.get(num_samples_to_process)
            yield chunk


async def _discard_results(
    _: ProcessConfig,
    queue: asyncio.Queue,
) -> None:
    """
    discard all results from async queue
    """

    while True:
        _ = await queue.get()
        queue.task_done()


async def scan_for_frequencies(
    source_gen: AsyncIterator[np.ndarray],
    process_config: ProcessConfig,
) -> list[float]:

    assert process_config.carrier_freq is None

    try:
        frequencies = await find_beep_frequencies(source_gen, process_config, N=13)

        if not frequencies:
            logger.error("No frequency detected, exiting...")
            raise CarrierFrequencyNotFound()
        else:
            logger.info(f"Frequencies detected: {frequencies} - end scanning...")

        return frequencies

    except CarrierFrequencyNotFound:
        logger.exception("Carrier frequency not found, interrupting sample processing...")
        raise


async def queue_to_iterator(q: asyncio.Queue):
    while True:
        obj = await q.get()
        q.task_done()
        yield obj


async def scan_frequencies_background(
    input_queue: asyncio.Queue,
    output_queue: asyncio.Queue,
    signal_scanning_event: asyncio.Event,
    process_config: ProcessConfig,
):
    process_config = deepcopy(process_config)
    process_config.carrier_freq = None

    if process_config.sample_config.scan_interval is None or process_config.sample_config.scan_interval == 0:
        logger.info("Background scan interval task is disabled.")
        while True:
            await asyncio.sleep(3600)

    minutes = process_config.sample_config.scan_interval

    while True:
        logger.info(f"scan_interval_task start sleeping for {minutes * 60} seconds...")
        await asyncio.sleep(minutes * 60)
        logger.info("scan_interval_task woke up, start scanning...")

        # signal that we want to receive samples
        signal_scanning_event.set()

        try:
            frequencies = await scan_for_frequencies(queue_to_iterator(input_queue), process_config)
            logger.info(f"frequencies found {frequencies=}")
            output_queue.put_nowait(frequencies)
        except CarrierFrequencyNotFound:
            logger.info("No frequencies found, using old ones.")

        # we're done with receiving samples
        signal_scanning_event.clear()

        # clear leftover samples in queue
        while not input_queue.empty():
            _ = await input_queue.get()
            input_queue.task_done()


def create_structures_from_frequencies(
    frequencies: list[float],
    pc: ProcessConfig,
    executor: ProcessPoolExecutor,
    task_results,
):
    # for each detected/defined frequency we create two tasks:
    # - task for processing sample
    # - task for handling processed sample (detected BPM.)
    #
    # for each `processing sample task` we create `process queue` and `result_queue`
    # so we can distribute samples from task_samples_source() to each `process queue`
    # `processing sample task` will put detected BPMs to `result_queue` for further handling.
    process_queues, result_queues, process_tasks, result_tasks = set(), set(), set(), set()
    # for i, pc in enumerate(process_config, 1):
    for i, f in enumerate(frequencies, 1):
        logger.debug(f"Creating sample process task and results task no.{i}, frequency={f}")

        p = deepcopy(pc)
        p.carrier_freq = f

        result_queue = asyncio.Queue()
        process_queue = asyncio.Queue(maxsize=3)

        task_sample_processor = asyncio.create_task(
            process_sample(
                p,
                executor,
                process_queue,
                [result_queue],
            )
        )
        task_result = asyncio.create_task(task_results(p, result_queue))

        process_queues.add(process_queue)
        result_queues.add(result_queue)

        process_tasks.add(task_sample_processor)
        result_tasks.add(task_result)

    return process_queues, result_queues, process_tasks, result_tasks


async def process_results(
    pc: ProcessConfig,
    queue_results: asyncio.Queue,
    task_results,
) -> None:
    # for each channel we will have separate async task
    queues = {}

    tasks = set()

    while True:
        res: ProcessResult = await queue_results.get()

        q = queues.get(res.channel, None)
        if not q:
            p = deepcopy(pc)
            p.carrier_freq = res.carrier_freq
            q = asyncio.Queue()
            task_result = asyncio.create_task(task_results(p, q))
            tasks.add(task_result)
            queues[res.channel] = q

        await q.put(res)

        queue_results.task_done()


async def pipeline(
    process_config: list[ProcessConfig] | ProcessConfig,
    source_gen: AsyncIterator[np.ndarray],
    task_results=None,
) -> None:
    """
    process_config     -> ...
    task_samples_input -> callable with one argument (samples_queue), must return async task
    task_results       -> callable with one argument (out_queue), must return async task
                          can be None (then all results are discarded)
    """

    if task_results is None:
        task_results = _discard_results

    # # what process config we have?
    # # - list of process configs
    # # - single process config with carrier_freq=None (should scan.)
    # # - single process config with defined carrier_freq
    # match process_config:
    #     case list():
    #         # TODO: process config could be list (define carrier frequencies from command line)
    #         # list of fully defined ProcessConfigs (with carrier_freq defined from command line)
    #         raise NotImplementedError("Multiple process_configs in pipeline not yet implemented.")
    #     case ProcessConfig() if process_config.carrier_freq is None:
    #         # carrier_freq is None - we should scan for frequencies
    #         logger.info("Carrier frequency not set - start scanning...")
    #         frequencies = await scan_for_frequencies(source_gen, process_config)

    #         # TODO: create multime process sample tasks, each with different process_config, frequency, queues...
    #         # logger.info(f"Picking first one: {frequencies[0]}")
    #         # process_config.carrier_freq = frequencies[0]
    #         # process_config.carrier_freq = 160270968
    #         # process_config = [process_config]
    #     case ProcessConfig():
    #         # fully defined ProcessConfig (with carrier_freq)
    #         # process_config = [process_config]
    #         frequencies = [process_config.carrier_freq]
    #     case _:
    #         raise ValueError(f"Type of process_config {type(process_config)} not understood.")

    queue_input = asyncio.Queue()
    queue_output = asyncio.Queue()

    process_result_task = asyncio.create_task(
        process_results(
            process_config,
            queue_output,
            task_results,
        )
    )

    t = asyncio.Task(process_sample_new(process_config, queue_input, queue_output))

    async for sample in source_gen:
        await queue_input.put(sample)

    return
    ####################################################################################################################

    # setup background scanning:
    bg_scan_interval_input_queue = asyncio.Queue()
    bg_scan_interval_output_queue = asyncio.Queue()
    bg_scan_interval_event = asyncio.Event()

    task_scan_interval = asyncio.create_task(
        scan_frequencies_background(
            bg_scan_interval_input_queue,
            bg_scan_interval_output_queue,
            bg_scan_interval_event,
            process_config,
        )
    )

    # pr = cProfile.Profile()
    # pr.enable()

    with ProcessPoolExecutor(max_workers=4) as executor:
        while True:
            process_queues, result_queues, process_tasks, result_tasks = create_structures_from_frequencies(
                frequencies,
                process_config,
                executor,
                task_results,
            )

            # snapshot1 = tracemalloc.take_snapshot()

            # distribute samples to all process queues:
            async for sample in source_gen:
                for pq in process_queues:
                    await pq.put(sample.copy())

                # is background scanning active?
                # If yes, put sample to the scan_frequncies queue...
                if bg_scan_interval_event.is_set():
                    await bg_scan_interval_input_queue.put(sample.copy())
                # has background scanning produced new frequencies?
                elif not bg_scan_interval_output_queue.empty():
                    frequencies = await bg_scan_interval_output_queue.get()
                    bg_scan_interval_output_queue.task_done()
                    logger.info(f"New frequencies found: {frequencies=}, creating new structrures...")
                    break

                # snapshot2 = tracemalloc.take_snapshot()
                # top_stats = snapshot2.compare_to(snapshot1, "lineno")

                # logger.info("[ Top 10 differences ]")
                # for stat in top_stats[:10]:
                #     logger.info(stat)

            else:
                break  # `source_gen` is exhausted, end all processing

        # wait for processing:
        for pq in [*process_queues, *result_queues, bg_scan_interval_input_queue]:
            await pq.join()

        # cancel all tasks:
        for t in [*process_tasks, *result_tasks, task_scan_interval]:
            t.cancel()

    # pr.disable()

    # s = io.StringIO()
    # sortby = pstats.SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())


if __name__ == "__main__":
    main()
