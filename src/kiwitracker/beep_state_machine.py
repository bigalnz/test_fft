from kiwitracker.chicktimer import ChickTimer
from kiwitracker.common import ProcessConfig
import datetime
import math

from kiwitracker.chicktimer import ChickTimer
from kiwitracker.common import ProcessConfig
from datetime import datetime
import math



# 3 sec gap == 20BPM          ( marks start of CT sequence and after each 5 beep seperator )
# 3.8 sec gap == 15.789BPM    ( after each number in number pairs )
# 1.3 sec gap == 46.1538BPM   ( between each beep of 5 beep seperators )
# 0.8 sec gap == 75.00BPM     ( between each beep of data beeps )

class BeepStateMachine:

    gap_beep_rate_3sec: float = 20.00
    gap_beep_rate_3_8sec: float = 15.789
    gap_beep_rate_1_3sec: float = 46.153
    gap_beep_rate_0_8: float = 75.00

    config: ProcessConfig

    number1_count: int
    number2_count: int
    seperator_count: int
    pair_count: int

    def __init__(self, config: ProcessConfig, inital_state:str = "BACKGROUND"):
        self.state = inital_state
        self.number1_count = 1
        self.number2_count = 1
        self.seperator_count = 1
        self.pair_count = 0
        self.ct = ChickTimer()
    
    @property
    def channel(self): 
        """Channel Number from Freq"""
        return math.floor((self.config.carrier_freq - 160.11e6)/0.01e6)
    
    @property
    def carrier_freq(self):
        """Center frequency of the carrier wave to process (in Hz)"""
        return self.config.carrier_freq
        self.config = config
    
    @property
    def channel(self): 
        """Channel Number from Freq"""
        return math.floor((self.config.carrier_freq - 160.11e6)/0.01e6)
    
    @property
    def carrier_freq(self):
        """Center frequency of the carrier wave to process (in Hz)"""
        return self.config.carrier_freq
        
    def process_input(self, BPM: float) -> None|ChickTimer:
        if self.state == "BACKGROUND":
            if any(abs(BPM-background_beep_rate) < 0.5 for background_beep_rate in [80, 46, 30] ):
                # background beep rate, do nothing, return nothing, exit
                return
            if (abs(BPM - self.gap_beep_rate_3sec) < 0.5):
                # 3 secon pause encountered - indicates first set of digits
                # change state and record carrier freq and channel No
                # record start date time
                # change state and record carrier freq and channel No
                # record start date time
                # change state and record carrier freq and channel No
                # record start date time
                self.state = "NUMBER1"
                self.ct.channel = self.channel
                self.ct.start_date_time = datetime.now()

                self.ct.channel = self.channel
                self.ct.carrier_freq = self.carrier_freq
                print(f" ************ CT start ***********")
                self.ct.start_date_time = datetime.now()
                return
    
        if self.state == "NUMBER1":
            # check expected BPM and if so count and increment
            if (abs(BPM - self.gap_beep_rate_0_8) < 2.5 ):
                self.number1_count += 1
                return
            # if BPM is 15.78 - exit as that was last beep of the set
            if (abs(BPM - self.gap_beep_rate_3_8sec ) < 1.0):
                self.state = "NUMBER2"
                return # this return needs to exit both loops?
       
        if self.state == "NUMBER2":
            if (abs(BPM - self.gap_beep_rate_0_8) < 2.5 ):
                self.number2_count += 1
                return
            # if BPM is 15.78 - exit as last beep was last beep of that set
            if (abs(BPM - self.gap_beep_rate_3_8sec ) < 0.5):
                self.ct.setField(self.pair_count, int(f"{self.number1_count}{self.number2_count}" ) )
                print(f"CT so far : {self.ct} ")
                self.number1_count = 1
                self.number2_count = 1
                if (self.pair_count == 7): # exit if last pair
                    self.state = "FINISHED"
                    return
                self.state = "SEPERATOR"
                return
            
        if self.state == "SEPERATOR":
            if (abs(BPM - self.gap_beep_rate_1_3sec) < 1.0):
                self.seperator_count += 1
                return
            if (self.seperator_count == 5): # check for 5 beeps in seperator - should I also check that the gap is 3s?
                self.seperator_count = 0
                self.state = "NUMBER1"
                self.pair_count += 1
                return
            if self.seperator_count > 5: 
                print(f"Seperator count exceeded 5 - returning to background")
                # reset everything for early bsm exit
                self.state == "BACKGROUND"
                self.number1_count = 1
                self.number2_count = 1
                self.seperator_count = 1
                return


        
        # Check we have 8 pairs of numbers and a 3 sec end pause
        if self.state == "FINISHED":
            print(f"*********** CT's have been recorded : {self.ct} **************")
            self.number1_count = 1
            self.number2_count = 1
            self.seperator_count = 1
            self.pair_count = 0
            self.state = "BACKGROUND"
            return