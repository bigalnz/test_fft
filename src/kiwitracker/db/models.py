from sqlalchemy import Column, DateTime, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
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
