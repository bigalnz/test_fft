import datetime as dt
from dataclasses import dataclass
from typing import TypedDict
from enum import Enum

@dataclass
class ChickTimerStatus():
    DaysSinceStatusChanged: int
    DaysSinceHatch: int
    DaysSinceDesertionTriggered: int
    TimeOfEmergence: int
    WeeksOfBatteryLeft: int
    ActivityYesterday: int
    ActivityTwoDaysAgo: int
    MeanActivity: int

    def getValues(self):
        return [self.DaysSinceStatusChanged, self.DaysSinceHatch, \
                self.DaysSinceDesertionTriggered, self.TimeOfEmergence, \
                self.WeeksOfBatteryLeft, self.ActivityYesterday, \
                self.ActivityTwoDaysAgo, self.MeanActivity]
    
    # Convert ChickTimerStatus to a dict, able to access by index of key?
    def setValue(self, fieldIndex, value):
            match fieldIndex:
                case 0:
                      self.DaysSinceStatusChanged = value
                case 1:
                      self.DaysSinceHatch = value
                case 2:
                      self.DaysSinceDesertionTriggered = value
                case 3:
                      self.TimeOfEmergence = value
                case 4:
                      self.WeeksOfBatteryLeft = value
                case 5:
                      self.ActivityYesterday = value
                case 6:
                      self.ActivityTwoDaysAgo = value
                case 7:
                      self.MeanActivity = value
                case _:
                    print("* * *  ERROR: Unknown field index passed into ChickTimer Class * * *")

class ChickTimerMode(Enum):
    NotIncubating = 30
    Incubating = 48
    Mortality = 80

class ChickTimer():
    status : ChickTimerStatus
    channel : int
    mode: ChickTimerMode

    def __init__(self):
        self.status = ChickTimerStatus(
            DaysSinceStatusChanged = 0,
            DaysSinceHatch = 0,
            DaysSinceDesertionTriggered = 0,
            TimeOfEmergence = 0,
            WeeksOfBatteryLeft = 0,
            ActivityYesterday = 0,
            ActivityTwoDaysAgo = 0,
            MeanActivity = 0
        )
        self.channel = 0
        self.mode = ChickTimerMode.NotIncubating

    def __str__(self):
        return f'ChickTimer:\n\t{self.status}\n\tMode: {self.mode}\n\tChannel: {self.channel}'