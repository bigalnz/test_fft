from .chicktimer import ChickTimer

# 3 sec gap == 20BPM          ( marks start of CT sequence and after each 5 beep seperator )
# 3.8 sec gap == 15.789BPM    ( after each number in number pairs )
# 1.8 sec gap == 46.1538BPM   ( between each beep of 5 beep seperators )
# 0.8 sec gap == 75.00BPM     ( between each beep of data beeps )

class BeepStateMachine:

    gap_beep_rate_3sec: float = 20.00
    gap_beep_rate_3_8sec: float = 15.789
    gap_beep_rate_1_8sec: float = 46.13
    gap_beep_rate_0_8: float = 75.00


    def __init__(self, inital_state:str = "BACKGROUND"):
        self.state = inital_state

    def process_input(self, BPM: float) -> None|ChickTimer:
        if self.state == "BACKGROUND":
            if any(abs(BPM-background_beep_rate) < 0.5 for background_beep_rate in [80, 46, 30] ):
                # background beep rate, do nothing, return nothing, exit
                return
            if any(abs(BPM - self.gap_beep_rate_3sec) < 0.5):
                # 3 secon pause encountered - indicates first set of digits
                print("inside this block")
                self.state = "PAIR1_SET1"
    
        if self.state == "PAIR1_SET1":
            beep_count: int = 0
            # check expected BPM and if so count and increment
            if any(abs(BPM - self.gap_beep_rate_0_8) < 0.5 ):
                beep_count += 1
                return
            # if BPM is 15.78 - exit as that was last beep of the set
            if any(abs(BPM - self.gap_beep_rate_3_8sec ) < 0.5):
                # write the number of beeps to the data class
                print(f"Pair 1 Set 1 : {beep_count}")
                beep_count = 0
                self.state == "PAIR1_SET2"
                return # this return needs to exit both loops?
       
        if self.state == "PAIR1_SET2":
            beep_count: int = 0
            if any(abs(BPM - self.gap_beep_rate_0_8) < 0.5 ):
                beep_count += 1
                return
            # if BPM is 15.78 - exit as last beep was last beep of that set
            if any(abs(BPM - self.gap_beep_rate_3_8sec ) < 0.5):
                # write the number of beeps to the data class
                print(f"Pair 1 Set 2 : {beep_count}")
                beep_count = 0
                self.state = "SEPERATOR"
                return # this return needs to exit both loops?
            
        if self.state == "SEPERATOR":
            beep_count = 0
            if any(abs(BPM - self.gap_beep_rate_1_8sec) < 0.5):
                beep_count += 1
                return
            if (beep_count == 5):
                self.state = "PAIR2_SET1" # BUT IT COULD BE PAIR 2,3,4,5,6,7,8 - HOW DO I DETERMINE AT THIS POINT?
                return
            


        



            