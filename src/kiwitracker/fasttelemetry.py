import datetime as datetime
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict


class FastTelemetryMode(Enum):
    NotNesting = 70
    Nesting = 40
    Mortality = 10
    Hatch = 100
    DesertionAlert = 130


@dataclass
class FastTelemetry:
    Mode: FastTelemetryMode
    DaysSinceStatusChanged: int
    WeeksOfBatteryLeft: int
    MeanActivity: int
    ActivityYesterday: int

    def __init__(self):
        self.Mode = None
        self.DaysSinceStatusChanged = None
        self.WeeksOfBatteryLeft = None
        self.MeanActivity = None
        self.ActivityYesterday = None


FT_DICT = {
    80.0: {
        0: 80.0,
        "Mortality": 78.95,
        1: 77.92,
        2: 76.92,
        "Nesting": 75.95,
        3: 75.0,
        4: 74.07,
        "Not Nesting": 73.17,
        5: 72.29,
        6: 71.43,
        "Hatch": 70.59,
        7: 69.77,
        8: 68.97,
        "Deserting": 68.18,
        9: 67.42,
    },
    48.0: {
        0: 48.0,
        "Mortality": 47.62,
        1: 47.24,
        2: 46.88,
        "Nesting": 46.51,
        3: 46.15,
        4: 45.8,
        "Not Nesting": 45.45,
        5: 45.11,
        6: 44.78,
        "Hatch": 44.44,
        7: 44.12,
        8: 43.8,
        "Deserting": 43.48,
        9: 43.17,
    },
    34.28: {
        0: 34.28,
        "Mortality": 34.09,
        1: 33.89,
        2: 33.7,
        "Nesting": 33.51,
        3: 33.33,
        4: 33.14,
        "Not Nesting": 32.96,
        5: 32.78,
        6: 32.6,
        "Hatch": 32.43,
        7: 32.25,
        8: 32.08,
        "Deserting": 31.91,
        9: 31.74,
    },
    30.0: {
        0: 30.0,
        "Mortality": 29.85,
        1: 29.7,
        2: 29.56,
        "Nesting": 29.41,
        3: 29.27,
        4: 29.13,
        "Not Nesting": 28.99,
        5: 28.85,
        6: 28.71,
        "Hatch": 28.57,
        7: 28.44,
        8: 28.3,
        "Deserting": 28.17,
        9: 28.04,
    },
    20.0: {
        0: 20.0,
        "Mortality": 19.93,
        1: 19.87,
        2: 19.8,
        "Nesting": 19.74,
        3: 19.67,
        4: 19.61,
        "Not Nesting": 19.54,
        5: 19.48,
        6: 19.42,
        "Hatch": 19.35,
        7: 19.29,
        8: 19.23,
        "Deserting": 19.17,
        9: 19.11,
    },
    16.0: {
        0: 16.0,
        "Mortality": 15.96,
        1: 15.92,
        2: 15.87,
        "Nesting": 15.83,
        3: 15.79,
        4: 15.75,
        "Not Nesting": 15.71,
        5: 15.67,
        6: 15.63,
        "Hatch": 15.58,
        7: 15.54,
        8: 15.5,
        "Deserting": 15.46,
        9: 15.42,
    },
}
