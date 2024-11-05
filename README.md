# kiwitracker

A project to log to standard console the beeps per minute (BPM) of a Chick Timer (CT) for New Zealands endangered Kiwi. The console output will contain background BPM, Fast Telemetry (FT) numbers, Chick Timer (CT) data sequence, signal to noise ration (SNR), decibels full scale (dBFS) and finally Latitude/Longitude.

Currently the only way to decode these signals is to travel to remote parts of the bush and manually record the signal output. This is time consuming and if a predator is killing kiwi it may not be detected for several weeks. Kiwitracker allows for real time remote monitoring of signals and more granular data.

It is also possible to drone mount the hardware and geo-locate Kiwi quickly by filtering GPS and dBFS data.

## Kiwi Tracker Signals

Kiwi trackers emit 10mW continuous wave (CW) beeps on one of 100 channels spaced evenly between 160.120Mhz and 161.110Mhz. A list of the channel numbers and corresponding frequencies is [here](https://github.com/bigalnz/test_fft/blob/main/src/kiwitracker/freq_chart.txt).

Each beep is 0.017 sec long. By varying the time between beeps information is encoded in the signal in three different ways: 

## Modes of tranmission

### 1. Background Beep Rate mode

The transmitter is in this mode by default. By counting the number of beeps per minute the state of the bird can be determined:

* 80 BPM - Mortality mode
* 46-48 BPM - Not incubating
* 30 BPM - Incubating

### 2. Chick Timer (CT) mode

  Every 10 minutes the transmitter pauses for 3 seconds then emits 8 groups of two digit numbers. Each pair of two digit numbers is seperated by a 3 second pause.

  1. Days since change of state
  2. Days since hatch
  3. Days since desertion alert
  4. Time of emergence
  5. Weeks of batt life for Tx
  6. Activity yesterday
  7. Activity 2 days ago
  8. True mean of last 4 days (activity)

### 3. Fast Telemetry (FT) mode

Modes 1 and 2 are easily decoded by listening to the received signal. FT mode can only be decoded by computer as it involved altering by 10ms the timing between beeps in both modes 1 and 2. 

The alteration in timing decodes to:

Fast Telemetry table:

| Encoded Value | Pulse Delay(ms) | BPM       |       |       |       |       |       |
|---------------|-----------------|-----------|-------|-------|-------|-------|-------|
| 0             | 0               | 80        | 48    | 34.28 | 30    | 20    | 16    |
| Mortality     | 10              | 78.95     | 47.62 | 34.09 | 29.85 | 19.93 | 15.96 |
| 1             | 20              | 77.92     | 47.24 | 33.89 | 29.70 | 19.87 | 15.92 |
| 2             | 30              | 76.92     | 46.88 | 33.70 | 29.56 | 19.80 | 15.87 |
| Nesting       | 40              | 75.95     | 46.51 | 33.51 | 29.41 | 19.74 | 15.83 |
| 3             | 50              | 75.00     | 46.15 | 33.33 | 29.27 | 19.67 | 15.79 |
| 4             | 60              | 74.07     | 45.80 | 33.14 | 29.13 | 19.61 | 15.75 |
| Not Nesting   | 70              | 73.17     | 45.45 | 32.96 | 28.99 | 19.54 | 15.71 |
| 5             | 80              | 72.29     | 45.11 | 32.78 | 28.85 | 19.48 | 15.67 |
| 6             | 90              | 71.43     | 44.78 | 32.60 | 28.71 | 19.42 | 15.63 |
| Hatch         | 100             | 70.59     | 44.44 | 32.43 | 28.57 | 19.35 | 15.58 |
| 7             | 110             | 69.77     | 44.12 | 32.25 | 28.44 | 19.29 | 15.54 |
| 8             | 120             | 68.97     | 43.80 | 32.08 | 28.30 | 19.23 | 15.50 |
| Deserting     | 130             | 68.18     | 43.48 | 31.91 | 28.17 | 19.17 | 15.46 |
| 9             | 140             | 67.42     | 43.17 | 31.74 | 28.04 | 19.11 | 15.42 |

Full manual http://wildtech.co.nz/downloads/NiB%20CT%20V3.4.pdf


## To Do:

- [X] Add samples for testing purposes (weak signals, signals on ch00 and ch99, multiple signals, signals at 30, 48, 80 bpm, CT signals)
- [X] In sample_processor handle edge case where chunk edge goes through middle of a beep (falling edge array is empty). Add Boolean 'no_falling_edge' and track number of high samples for calculation of `BEEP_DURATION` on next set of chunks.
- [X] Lower threshold to more suitable value - from 0.9 to ?
- [ ] Test performance on Rpi4
- [ ] Add sample signal on Ch00 and Ch99 for Nyquist edge test
- [ ] Add sample signal for 30BPM (incubation mode)
- [X] Add validation (1) background BPM output with tolerance +/- 1? `[expected for expected in [80, 46, 47, 48, 30] if abs(actual - expected) < 1]` and (2) BEEP_DURATION must be 0.017 sec
- [X] Processing of CT signals - start with a data class for CT and then CT detection (3 second pause in gap between beeps, i.e. rising_edges)
- [X] Add SNR output for each beep
- [ ] How often to do a channel scan? Hourly?
- [ ] Add option to log signals to MySQL
- [ ] Change Fc default to closer to being between middle freqs (i.e. 160.625Mhz) if bandwidth is 1.5Mhz. Else if bandwidth is 2.048Mhz then Fc can be out of band for freqs of interest - performance question on Rpi4 / No need to deal with DC spike at Fc.
- [ ] Scan and Log CT signals daily.

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

## Usage

Once installed, the project can be invoked on the command line:

```bash
kiwitracker --help
```

Which should display the following:

```bash
usage: kiwitracker [-h] [-f INFILE] [-o OUTFILE] [-m MAX_SAMPLES] [-c CHUNK_SIZE] [-s SAMPLE_RATE] [--center-freq CENTER_FREQ] [-g GAIN] [--carrier CARRIER]

options:
  -h, --help            show this help message and exit

  -f INFILE, --from-file INFILE
                        Read samples from the given filename and process them

  -o OUTFILE, --outfile OUTFILE
                        Read samples from the device and save to the given filename

  -m MAX_SAMPLES, --max-samples MAX_SAMPLES
                        Number of samples to read when "-o/--outfile" is specified

Sampling:
  -c CHUNK_SIZE, --chunk-size CHUNK_SIZE
                        Chunk size for sdr.read_samples (default: 65536)

  -s SAMPLE_RATE, --sample-rate SAMPLE_RATE
                        SDR sample rate (default: 1024000.0)
  --center-freq CENTER_FREQ
                        SDR center frequency (default: 160270968)

  -g GAIN, --gain GAIN  SDR gain (default: 7.7)

Processing:
  --carrier CARRIER     Carrier frequency to process (default: 160707760)
```
