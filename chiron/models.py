from sqlalchemy import Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class CacheEntry(Base):
    __tablename__ = "cache"
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, nullable=True)
    file_path = Column(String, nullable=True)
