import datetime as datetime
from dataclasses import dataclass, field
from typing import TypedDict
from enum import Enum
import json

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

@dataclass
class SignalToNoiseRatio():
     min: float
     max: float
     mean: float
@dataclass
class DecibelsFullScale():
      min: float
      max: float
      mean: float
     
class ChickTimer():
      status : ChickTimerStatus
      channel : int
      mode: ChickTimerMode
      status : ChickTimerStatus
      channel : int
      mode: ChickTimerMode
      start_date_time : datetime.date
      finish_date_time : datetime.date
      snr : SignalToNoiseRatio
      dbfs : DecibelsFullScale
      lat : float = 0
      lon : float = 0
      carrier_freq : float = 0

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
        self.carrier_freq = 0
        self.lat = 0
        self.lon = 0
        self.start_date_time = datetime.datetime(1900, 1, 1)
        self.finish_date_time = datetime.datetime(1900, 1, 1)
        self.dbfs = DecibelsFullScale(
            min = 0.0,
            max = 0.0,
            mean = 0.0
        )
        self.snr = SignalToNoiseRatio(
             min = 0.0,
             max = 0.0,
             mean = 0.0
        )
        self.mode = ChickTimerMode.NotIncubating

      def __str__(self):
        return f'ChickTimer:\n\t{self.status}\n\tMode: {self.mode}\n\tChannel: {self.channel}'
      
      def toJSON(self):
        return json.dumps({"start_date_time" : self.start_date_time.strftime("%Y%m%d-%H%M%S"), \
                               "channel" : self.channel, \
                                 "snr" : { \
                                      "min" : self.snr.min, \
                                      "max" : self.snr.max, \
                                      "mean" : self.snr.mean \
                                 }, \
                                 "dbfs" : { \
                                      "min" : self.dbfs.min, \
                                      "max" : self.dbfs.max, \
                                      "mean" : self.dbfs.mean \
                                 }, \
                                 "lat" : self.lat, \
                                 "lon" : self.lon, \
                                 "carrier_freq" :  self.carrier_freq, \
                                 "days_since_change_of_state" : self.status.DaysSinceStatusChanged, \
                                 "days_since_hatch" : self.status.DaysSinceHatch, \
                                 "days_since_desertion_alert" : self.status.DaysSinceDesertionTriggered, \
                                 "time_of_emergence" : self.status.TimeOfEmergence, \
                                 "weeks_batt_life_left" : self.status.WeeksOfBatteryLeft, \
                                 "activity_yesterday" : self.status.ActivityYesterday, \
                                 "activity_two_days_ago" : self.status.ActivityTwoDaysAgo, \
                                 "mean_activity_last_four_days" : self.status.MeanActivity, \
                                 "finish_date_time" : self.finish_date_time.strftime("%Y%m%d-%H%M%S")}, \
                                indent = 4)