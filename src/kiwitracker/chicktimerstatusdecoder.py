from kiwitracker.chicktimerv2 import ChickTimer, ChickTimerMode


## Function Annotation to initialize each coroutine
def prime(fn):
    def wrapper(*args, **kwargs):
        v = fn(*args, **kwargs)
        v.send(None)
        return v

    return wrapper


# Finite State Machine decoder for the Chick Timer main signal
class ChickTimerStatusDecoder:
    def __init__(self):
        self.background = self._create_background()
        self.tens_digit = self._create_tens_digit()
        self.ones_digit = self._create_ones_digit()
        self.seperator = self._create_seperator()
        self.current_state = self.background

        self.ct = ChickTimer()
        self.hasValidChickTimer = False
        self.digitAccumulator = 0
        self.digitZeroOffset = 2
        self.backgroundCounter = 0
        self.fieldIndex = 0
        self.seperatorCounter = 0

    def send(self, bpm):
        # print(self.current_state)
        self.current_state.send(bpm)

    def getChickTimer(self):
        return ct

    def hasValidChickTimerStatus(self):
        return self.hasValidChickTimer

    def __str__(self):
        return f"State: {self.current_state}\tField: {self.fieldIndex}\tAccumulator: {self.digitAccumulator}"  # \nChickTimer: {ct}'

    @prime
    def _create_background(self):
        while True:
            bpm = yield
            if bpm in [30, 48, 80]:
                self.current_state = self.background
                self.ct.mode = ChickTimerMode(bpm)
                self.backgroundCounter += 1
                if self.backgroundCounter > 10 and self.fieldIndex != 0:
                    self.hasValidChickTimer = False
            elif bpm in [20]:
                self.current_state = self.tens_digit
                self.digitAccumulator += 1
            elif bpm in [16]:
                self.current_state = self.ones_digit
                self.digitAccumulator += 1

    @prime
    def _create_tens_digit(self):
        while True:
            bpm = yield
            if bpm in [80]:
                self.current_state = self.tens_digit
                self.digitAccumulator += 1
            elif bpm in [16]:
                self.current_state = self.ones_digit
                self.digitAccumulator = (self.digitAccumulator - self.digitZeroOffset) * 10
                self.digitAccumulator += 1

    @prime
    def _create_ones_digit(self):
        while True:
            bpm = yield
            if bpm in [80]:
                self.current_state = self.ones_digit
                self.digitAccumulator += 1
            elif bpm in [16]:
                self.ct.status.setValue(self.fieldIndex, self.digitAccumulator - self.digitZeroOffset)
                self.digitAccumulator = 0
                if self.fieldIndex == 7:
                    self.current_state = self.background
                    self.hasValidChickTimer = True
                    self.fieldIndex = 0
                else:
                    self.current_state = self.seperator
                    self.fieldIndex += 1

    @prime
    def _create_seperator(self):
        while True:
            bpm = yield
            if bpm in [48, 80]:
                if self.seperatorCounter > 5:
                    self.hasValidChickTimer = False
                    self.current_state = self.background
                    self.seperatorCounter = 0
                else:
                    self.seperatorCounter += 1
            if bpm in [20]:
                self.current_state = self.tens_digit
                self.seperatorCounter = 0
                self.digitAccumulator += 1
