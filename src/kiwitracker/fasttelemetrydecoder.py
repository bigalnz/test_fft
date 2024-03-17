from kiwitracker.fasttelemetry import FastTelemetry, FastTelemetryMode


class FastTelemetryDecoder:
    def __init__(self):
        self.bpm = 0
        self.positional_counter = 0
        self.digit = 0
        self.mode = None
        self.fast_telemetry = FastTelemetry()
        self.valid_intervals = [250, 750, 1250, 1750, 2000, 3000, 3750]
        self.fast_telemetry_dict = dict(
            {
                0: 0,
                10: FastTelemetryMode.Mortality,
                20: 1,
                30: 2,
                40: FastTelemetryMode.Nesting,
                50: 3,
                60: 4,
                70: FastTelemetryMode.NotNesting,
                80: 5,
                90: 6,
                100: FastTelemetryMode.Hatch,
                110: 7,
                120: 8,
                130: FastTelemetryMode.DesertionAlert,
                140: 9,
            }
        )

    def send(self, bpm):
        bpms = (60 / bpm) * 1000
        telemetry_interval = bpms - min(self.valid_intervals, key=lambda x: abs(x - bpms))
        value = self.fast_telemetry_dict[min(self.fast_telemetry_dict, key=lambda x: abs(x - telemetry_interval))]

        if value in [
            FastTelemetryMode.Mortality,
            FastTelemetryMode.Nesting,
            FastTelemetryMode.NotNesting,
            FastTelemetryMode.Hatch,
            FastTelemetryMode.DesertionAlert,
        ]:
            self.fast_telemetry.Mode = value
            self.positional_counter += 1
        elif self.positional_counter == 1:
            self.digit = value * 10
            self.positional_counter += 1
        elif self.positional_counter == 2:
            self.fast_telemetry.DaysSinceStatusChanged = self.digit + value
            self.positional_counter += 1
        elif self.positional_counter == 3:
            if self.fast_telemetry.Mode in [FastTelemetryMode.Mortality, FastTelemetryMode.NotNesting]:
                self.digit = value * 10
                self.positional_counter += 1
            elif self.fast_telemetry.Mode in [FastTelemetryMode.Nesting, FastTelemetryMode.Hatch]:
                self.digit = value * 10
                self.positional_counter += 1
            elif self.fast_telemetry.Mode in [FastTelemetryMode.DesertionAlert]:
                self.digit = value * 10
                self.positional_counter += 1
        elif self.positional_counter == 4:
            if self.fast_telemetry.Mode in [FastTelemetryMode.Mortality, FastTelemetryMode.NotNesting]:
                self.fast_telemetry.WeeksOfBatteryLeft = self.digit + value
                self.positional_counter = 0
                print(self.fast_telemetry)
            elif self.fast_telemetry.Mode in [FastTelemetryMode.Nesting, FastTelemetryMode.Hatch]:
                self.fast_telemetry.MeanActivity = self.digit + value
                self.positional_counter = 0
                print(self.fast_telemetry)
            elif self.fast_telemetry.Mode in [FastTelemetryMode.DesertionAlert]:
                self.fast_telemetry.ActivityYesterday = self.digit + value
                self.positional_counter = 0
                print(self.fast_telemetry)
