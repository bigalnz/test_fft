## Valid Beep Intervals

The valid nominal pulse intervals expressed in milliseconds (ms) and in BPM (beeps per minute) are:

    250ms   (240 BPM)
    750ms   (80 BPM) 
    1250ms  (48 BPM)
    1750ms  (34.29 BPM)
    2000ms  (30 BPM) 
    3000ms  ( 20 BPM)
    3750ms  ( 16 BPM)

However the encoding of fast telemetry between beeps means that the interval between beeps can vary. The fast telemetry pulse delays are:


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


For exampe this table show that the maximum value a 80 BPM can be is 80, while the minimum value for a 80 BPM is 67.41 BPM (and which would mean fast telemetry has transmitted a "9")
