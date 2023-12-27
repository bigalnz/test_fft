A project to log to standard console the BPM of a Chick Timer (CT) for Kiwi.

Suggested buffering chunk size -c 16384 or 65536 (to test)
Suggested sample rate -s 2.048e6 (allows covering from channel 00 (160.120Mhz), 01 (160.130).....99 (161.110Mhz) ) (Total Spectrum 0.990Mhz)
(Not configurable but FYI processing 2.56e5 samples at a time)

80 BPM - Mortality mode

46-48 BPM - Not incubating

30 BPM - 

**To Do:**
* Add samples for testing purposes
* Add SNR output for each beep
* How often to do a channel scan? Hourly?
* Add option to log signals to MySQL
* Change Fc to closer to being between middle freqs (i.e. 160.625Mhz) if bandwidth is 1.5Mhz. Else if bandwidth is 2.048Mhz then Fc can be out of band for freqs of interest - performance question on Rpi4.

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
