from kiwitracker.chicktimerv2 import ChickTimer, ChickTimerMode, ChickTimerStatus

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
        self.current_state = self.background

        self.ct = ChickTimer()
        self.hasValidChickTimer = False
        self.digitAccumulator = 0
        self.digitZeroOffset = 2
        self.backgroundCounter = 0
        self.fieldIndex = 0

    def send(self, bpm):
        self.current_state.send(bpm)

    def getChickTimer(self):
        return self.ct
    
    def hasValidChickTimerStatus(self):
        return self.hasValidChickTimer
    
    def __str__(self):
        return f'State: {self.current_state}\tField: {self.fieldIndex}\tAccumulator: {self.digitAccumulator}'#\nChickTimer: {ct}'
    
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
                self.current_state = self.background
                self.ct.status.setValue(self.fieldIndex, self.digitAccumulator - self.digitZeroOffset)
                self.digitAccumulator = 0
                if self.fieldIndex == 7:
                    self.hasValidChickTimer = True
                    self.fieldIndex = 0
                else:
                    self.fieldIndex += 1