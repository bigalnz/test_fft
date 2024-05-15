# kiwitracker

A project to log to standard console the BPM of a Chick Timer (CT) for Kiwi.

Suggested buffering chunk size `-c 16384` or `-c 65536` (to test)
Suggested sample rate `-s 2.048e6` (allows covering from channel 00 (160.120Mhz), 01 (160.130).....99 (161.110Mhz) ) (Total Spectrum 0.990Mhz)
(Not configurable but FYI processing 2.56e5 samples at a time)

## Background modes of transmitter

* 80 BPM - Mortality mode
* 46-48 BPM - Not incubating
* 30 BPM - Incubating

### Chick Timer (CT) modes

  Every 10 minutes the transmitter pauses for 3 seconds then emits 8 groups of two digit numbers. Each pair of two digit numbers is seperated by a 3 second pause.

  1. Days since change of state
  2. Days since hatch
  3. Days since desertion alert
  4. Time of emergence
  5. Weeks of batt life for Tx
  6. Activity yesterday
  7. Activity 2 days ago
  8. True mean of last 4 days (activity)

Full manual http://wildtech.co.nz/downloads/NiB%20CT%20V3.4.pdf


## To Do:

- [X] Add samples for testing purposes (weak signals, signals on ch00 and ch99, multiple signals, signals at 30, 48, 80 bpm, CT signals)
- [X] In sample_processor handle edge case where chunk edge goes through middle of a beep (falling edge array is empty). Add Boolean 'no_falling_edge' and track number of high samples for calculation of `BEEP_DURATION` on next set of chunks.
- [X] Lower threshold to more suitable value - from 0.9 to ?
- [X] Test performance on Rpi4
- [ ] Add sample signal on Ch00 and Ch99 for Nyquist edge test
- [ ] Add sample signal for 30BPM (incubation mode)
- [X] Add validation (1) background BPM output with tolerance +/- 1? `[expected for expected in [80, 46, 47, 48, 30] if abs(actual - expected) < 1]` and (2) BEEP_DURATION must be 0.017 sec
- [X] Processing of CT signals - start with a data class for CT and then CT detection (3 second pause in gap between beeps, i.e. rising_edges)
- [X] Add SNR output for each beep
- [ ] Add option to scan at X interval, and automatically change Fc to scan whole spectrum. i.e. "--scan 1" - scan once per hour "--scan 0.1" scan every 6 minutes, "--scan 24"" scan daily
- [ ] Add option to set an array of carriers --carrier [160707800, 160338000, .... ] (max 6 freqs)
- [X] Add option to support both RTLSDR and SDRPlay devices "--radio airspy" , "--radio rtl"
- [X] Add option to log signals to MySQL
- [X] Change Fc default to closer to being between middle freqs (i.e. 160.625Mhz) if bandwidth is 1.5Mhz. Else if bandwidth is 2.048Mhz then Fc can be out of band for freqs of interest - performance question on Rpi4 / No need to deal with DC spike at Fc.
- [X] Scan and Log CT signals daily - All CT signls, beeps and Fast Telemetry logged to seperate databases in SQLite.

## Sample Array Files

https://geekhelp-my.sharepoint.com/:f:/g/personal/al_geekhelp_co_nz/EhBFHUL_rwtKlAynyeeJMHoB3r3P6JYrl-5MQp0ihP7-_w?e=hoKMeB


## Installation

Installation within a [virtual environment](https://docs.python.org/3.11/library/venv.html) is highly recommended

### Pip

```bash
pip install 'kiwitracker@git+https://github.com/bigalnz/test_fft.git'
```

### Development Mode

#### Clone the repo

```bash
git clone git@github.com:bigalnz/test_fft.git
```

#### Install in "editable mode"

```bash
pip install -e .
```

#### Running the tests

```bash
pip install -e .[test]
```

and then from the project root:

```bash
pytest
```

## Usage

Once installed, the project can be invoked on the command line:

```bash
kiwitracker --help
```

Which should display the following:

```bash
usage: kiwitracker [-h] [-f INFILE] [-db DB] [-d] [-o OUTFILE] [-m MAX_SAMPLES] [--scan] [--no-use-gps] [--radio [{rtl,airspy}]] [-c CHUNK_SIZE] [-s SAMPLE_RATE] [--center-freq CENTER_FREQ] [-g GAIN]
                   [--bias-tee] [-log LOGLEVEL] [--carrier [CARRIER]]

options:
  -h, --help            show this help message and exit
  -f INFILE, --from-file INFILE
                        Read samples from the given filename and process them
  -db DB, --database DB
                        SQLite database where to store processed results. Defaults to `main.db`. Environment variable KIWITRACKER_DB has priority.
  -d, --delete-database
                        If SQLite database file exists upon start, it is deleted.
  -o OUTFILE, --outfile OUTFILE
                        Read samples from the device and save to the given filename
  -m MAX_SAMPLES, --max-samples MAX_SAMPLES
                        Number of samples to read when "-o/--outfile" is specified
  --scan                Scan for frequencies in first 3sec
  --no-use-gps          Set this flag to not use GPS module
  --radio [{rtl,airspy}]
                        type of radio to be used (default: rtl), ignored if reading samples from disk.

Sampling:
  -c CHUNK_SIZE, --chunk-size CHUNK_SIZE
                        Chunk size for sdr.read_samples (default: 65536)
  -s SAMPLE_RATE, --sample-rate SAMPLE_RATE
                        SDR sample rate (default: 1024000.0)
  --center-freq CENTER_FREQ
                        SDR center frequency (default: 160270968)
  -g GAIN, --gain GAIN  SDR gain (default: 7.7)
  --bias-tee            Enable bias tee
  -log LOGLEVEL, --loglevel LOGLEVEL
                        Provide logging level. Example --loglevel debug, default=warning

Processing:
  --carrier [CARRIER]   Carrier frequency to process (default: None)

```
