from abc import ABC, abstractmethod

import gpsd


class GPSBase(ABC):
    """
    Base abstract class for GPS module
    """

    def __init__(self):
        pass

    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def get_current(self) -> tuple[float, float]:
        pass


class GPSReal(GPSBase):
    """
    Real GPS module
    """

    def connect(self):
        gpsd.connect()

    def get_current(self):
        packet = gpsd.connect()
        return packet.lat, packet.lon


class GPSDummy(GPSBase):
    """
    Dummy GPS module to return fake data
    """

    def __init__(self, lat=-36.8807, lon=174.924):
        self.lat = lat
        self.lon = lon

    def connect(self):
        pass

    def get_current(self):
        return self.lat, self.lon
