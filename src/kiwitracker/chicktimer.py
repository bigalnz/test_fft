from dataclasses import dataclass
import datetime

@dataclass
class ChickTimer():

    start_date_time : datetime.date = 0
    finish_date_time : datetime.date = 0
    channel : int = 0
    carrier_freq : float = 0
    # eights sets of integers of two digits. Theese are the actual values not the transmitted values
    # so the 2 has already been subtracted.
    days_since_change_of_state : int = 0
    days_since_hatch : int = 0
    days_since_desertion_alert : int = 0
    # time in hours before now bird emerged for feeding
    time_of_emergence : int = 0
    weeks_batt_life_left : int = 0
    # mean actvity - multiply by 10 to get minutes active, i.e. 59 = 590 minutes = 9:50 (hh:mm)
    activity_yesterday : int = 0
    activity_two_days_ago : int = 0
    mean_activity_last_four_days : int = 0

    def setField(self, index: int, value: int):
        match index:
            case 0:
                self.days_since_change_of_state = value
            case 1: 
                self.days_since_hatch = value
            case 2: 
                self.days_since_desertion_alert = value
            case 3: 
                self.time_of_emergence = value
            case 4:
                self.weeks_batt_life_left = value
            case 5:
                self.activity_yesterday = value
            case 6:
                self.activity_two_days_ago = value
            case 7:
                self.mean_activity_last_four_days = value
            case _:
                print("* * *  ERROR: Unknown field index passed into ChickTimer Class * * *")

    