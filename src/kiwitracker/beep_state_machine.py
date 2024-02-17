from kiwitracker.chicktimer import ChickTimer
from kiwitracker.common import ProcessConfig
from datetime import datetime
import math
import pprint
import json
import numpy as np

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
        self.config = config
        self.snrs = list()
    
    @property
    def carrier_freq(self):
        """Center frequency of the carrier wave to process (in Hz)"""
        return self.config.carrier_freq
        
    @property
    def channel(self): 
        """Channel Number from Freq"""
        return math.floor((self.config.carrier_freq - 160.11e6)/0.01e6)
        
    def process_input(self, BPM: float, SNR: float, lat=0, lon=0) -> None|ChickTimer:
        if self.state == "BACKGROUND":
            if any(abs(BPM-background_beep_rate) < 2.5 for background_beep_rate in [80, 46, 30] ):
                # background beep rate, do nothing, return nothing, exit
                return
            if (abs(BPM - self.gap_beep_rate_3sec) < 2 ):
                # 3 secon pause encountered - indicates first set of digits
                # change state and record carrier freq and channel No
                # record start date time
                print(f"Got a 3 sec gap 20 BPM")
                self.state = "NUMBER1"
                self.ct.lon = lon
                self.ct.lat = lat
                self.ct.channel = self.channel
                self.ct.start_date_time = datetime.now()
                self.ct.carrier_freq = self.carrier_freq
                self.snrs.append(SNR)
                print(f" ************ CT start ***********")
                return
    
        if self.state == "NUMBER1":
            # check expected BPM and if so count and increment
            if (abs(BPM - self.gap_beep_rate_0_8) < 4 ): #75 BPM
                self.number1_count += 1
                self.snrs.append(SNR)
                #print(f"number 1 count : {self.number1_count}")
                return
            # if BPM is 15.78 - exit as that was last beep of the set
            if (abs(BPM - self.gap_beep_rate_3_8sec ) < 2.9): # 15 BPM
                print(f"number 1 finished")
                self.state = "NUMBER2"
                return # this return needs to exit both loops?
       
        if self.state == "NUMBER2":
            if (abs(BPM - self.gap_beep_rate_0_8) < 4 ): #75 BPM
                self.number2_count += 1
                #print(f"number 2 count : {self.number2_count}")
                self.snrs.append(SNR)
                return
            # if BPM is 15.78 - exit as last beep was last beep of that set
            if (abs(BPM - self.gap_beep_rate_3_8sec ) < 2.5): # 15 BPM
                print(f"number 2 finished")
                self.ct.setField(self.pair_count, int(f"{self.number1_count}{self.number2_count}" ) )
                print(f"CT so far : {self.ct} ")
                self.number1_count = 1
                self.number2_count = 1
                if (self.pair_count == 7): # exit if last pair
                    self.state = "FINISHED"
                    self.ct.finish_date_time = datetime.now()
                    return
                self.state = "SEPERATOR"
                return
            
        if self.state == "SEPERATOR":
            if self.seperator_count > 5:  # SEPERATOR COUNT EXCEED - SOMETHING WENT WRONG - ABORT
                print(f"Seperator count (46.153BPM 1.3s) exceeded 5 - returning to background")
                # reset everything for early bsm exit
                self.state = "BACKGROUND"
                self.number1_count = 1
                self.number2_count = 1
                self.seperator_count = 1
                self.ct = ChickTimer()
                print(f"state on exit {self.state}")
                return
            if (abs(BPM - self.gap_beep_rate_1_3sec) < 2.5): # 46 BPM 
                self.seperator_count += 1
                print(f"seperator count : {self.seperator_count}")
                return
            if (self.seperator_count == 5): # check for 5 beeps in seperator - should I also check that the gap is 3s?
                self.seperator_count = 1
                self.state = "NUMBER1"
                self.pair_count += 1
                return
            else:
                print(f"* * *  ERROR: Beep State Machine - SEPERATOR had no valid condition met * * *")
                # reset everything for early bsm exit
                self.state = "BACKGROUND"
                self.number1_count = 1
                self.number2_count = 1
                self.seperator_count = 1
                self.ct = ChickTimer()
                return


        # Check we have 8 pairs of numbers and a 3 sec end pause
        if self.state == "FINISHED":
            print([ np.min(self.snrs), np.mean(self.snrs), np.max(self.snrs) ])
            self.ct.snr = [np.min(self.snrs), np.mean(self.snrs), np.max(self.snrs) ]
            filename = 'captures/' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + 'Ch' + str(self.channel) + '.json'
            self.ct.toJSON()
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.ct.toJSON())
            self.number1_count = 1
            self.number2_count = 1
            self.seperator_count = 1
            self.pair_count = 0
            self.state = "BACKGROUND"
            self.ct = ChickTimer()
            return