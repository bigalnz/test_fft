from dataclasses import dataclass

@dataclass
class ChickTimer():

    # eights sets of integers of two digits. Theese are the actual values not the transmitted values
    # so the 2 has already been subtracted.

    days_since_change_of_state : int
    days_since_hatch : int
    days_since_desertion_alert : int
    # time in hours before now bird emerged for feeding
    time_of_emergence : int
    weeks_batt_life_left : int
    # mean actvity - multiply by 10 to get minutes active, i.e. 59 = 590 minutes = 9:50 (hh:mm)
    activity_yesterday : int
    activity_two_days_ago : int
    mean_activity_last_four_days : int

