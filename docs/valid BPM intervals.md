## Valid Beep Intervals

The valid nominal pulse intervals expressed in milliseconds (ms) and in BPM (beeps per minute) are:

    250ms   (240 BPM)
    750ms   (80 BPM) 
    1250ms  (48 BPM)
    1750ms  (34.29 BPM)
    2000ms  (30 BPM) 
    3000ms  ( 20 BPM)
    3750ms  ( 16 BPM)

However the encoding of fast telemtry between beeps means that the interval between beeps can vary. The fast telemetry pulse delays are:

| Pulse Delay (ms) | Encoded Value | 80 BPM | 48 BPM | 34.28 BPM | 30 BPM | 20 BPM | 16 BPM |
|------------------|---------------|--------|--------|-----------|--------|--------|--------|
| 0 | 0 | 80 | 48 |34.28 | 30 | 20 | 16 |
| 10 | Mortality |
| 20 | 1 |
| 30 | 2 |
| 40 | Nesting |
| 50 | 3 |
| 60 | 4 |
| 70 | Not Nesting |
| 80 | 5 | 
| 90 | 6 |
| 100 | Hatch | 
| 110 | 7 | 
| 120 | 8 |
| 130 | Deserting | 
| 140 | 9 | 67.41 | 43.16 | 31.74 | 28.03 | 19.1082 | 15.42 |

