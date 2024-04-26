from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class BPM(Base):

    __tablename__ = "bpm"

    bid = Column(Integer, primary_key=True, autoincrement=True)

    dt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    channel = Column(Integer, nullable=False)
    bpm = Column(Float, nullable=False)
    dbfs = Column(Float, nullable=False)
    clipping = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    snr = Column(Float, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)


class ChickTimerResult(Base):

    __tablename__ = "chick_timer"

    cid = Column(Integer, primary_key=True, autoincrement=True)

    channel = Column(Integer, nullable=False)
    carrier_freq = Column(Float, nullable=False)

    decoding_success = Column(Boolean, default=False, nullable=False)

    start_dt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    end_dt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    snr_min = Column(Float, nullable=False)
    snr_max = Column(Float, nullable=False)
    snr_mean = Column(Float, nullable=False)

    dbfs_min = Column(Float, nullable=False)
    dbfs_max = Column(Float, nullable=False)
    dbfs_mean = Column(Float, nullable=False)

    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    days_since_change_of_state = Column(Integer, nullable=True)
    days_since_hatch = Column(Integer, nullable=True)
    days_since_desertion_alert = Column(Integer, nullable=True)
    time_of_emergence = Column(Integer, nullable=True)
    weeks_batt_life_left = Column(Integer, nullable=True)
    activity_yesterday = Column(Integer, nullable=True)
    activity_two_days_ago = Column(Integer, nullable=True)
    mean_activity_last_four_days = Column(Integer, nullable=True)


class FastTelemetryResult(Base):

    __tablename__ = "fast_telemetry"

    fid = Column(Integer, primary_key=True, autoincrement=True)

    channel = Column(Integer, nullable=False)
    carrier_freq = Column(Float, nullable=False)

    start_dt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    end_dt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    snr_min = Column(Float, nullable=False)
    snr_max = Column(Float, nullable=False)
    snr_mean = Column(Float, nullable=False)

    dbfs_min = Column(Float, nullable=False)
    dbfs_max = Column(Float, nullable=False)
    dbfs_mean = Column(Float, nullable=False)

    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)

    mode = Column(String(16), nullable=False)
    d1 = Column(Integer, nullable=False)
    d2 = Column(Integer, nullable=False)
