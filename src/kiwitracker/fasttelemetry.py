import datetime as datetime
from dataclasses import dataclass, field
from typing import TypedDict
from enum import Enum
import json

class FastTelemetryMode(Enum):
    NotNesting = 70
    Nesting = 40
    Mortality = 10
    Hatch = 100
    DesertionAlert = 130

@dataclass
class FastTelemetry():
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
        










      
