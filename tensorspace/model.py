import os
import sys
import enum

from pathlib import Path
from sqlalchemy import (Boolean, Column, Date, Float, ForeignKey, Integer,
                        SmallInteger, String, LargeBinary, create_engine)
from sqlalchemy.types import Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tensorspace.util import get_data_home

Base = declarative_base()

def get_config_single_host(initialize=False):
    config = 'sqlite:///' + str(get_data_home() / 'index.db')
    engine = create_engine(config)
    session = sessionmaker(bind=engine)() 
    if initialize:
        Base.metadata.create_all(engine)
    return session

class ControlledImport:
    def __init__(self, session, has_files=False):
        self.home = get_data_home()
        if not self.home.is_dir():
            Path.mkdir(self.home, parents=True)
        if has_files:
            self.home = self.home / 'images'
            if not self.home.is_dir():
                Path.mkdir(self.home, parents=True)
        self.session = session

    def exists(self, fname):
        return (self.home / fname).exists()

    def path(self, fname):
        return str(self.home / fname)


class ImageFileType(enum.Enum):
    JPEG = 1
    PNG = 2
    GIF = 3
    TIFF = 4
    WEBP = 5


class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    filetype = Column(Enum(ImageFileType))
    src_filename = Column(String)
    url = Column(String)
    orig_url = Column(String)
    training_split = Column(Boolean(), nullable=False)
    license_id = Column(SmallInteger, ForeignKey(
        'licenses.id'), nullable=False)
    dataset_id = Column(SmallInteger, ForeignKey(
        'datasets.id'), nullable=False)
    captured = Column(Date)


class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    desc = Column(String)
    url = Column(String)


class Caption(Base):
    __tablename__ = 'captions'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    caption = Column(String)


class Object(Base):
    __tablename__ = 'objects'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    category_id = Column(Integer, ForeignKey('categories.id'), nullable=False)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)


class Category(Base):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    supercategory = Column(String)


class SegmentationCord(Base):
    __tablename__ = 'segmentationcords'
    id = Column(Integer, primary_key=True)
    cord = Column(Float)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False,
                      index=True)


class License(Base):
    __tablename__ = 'licenses'
    id = Column(SmallInteger, primary_key=True)
    url = Column(String)
    name = Column(String)

class Vector(Base):
    __tablename__ = 'vectors'
    id = Column(Integer, primary_key=True)
    caption_id = Column(Integer, ForeignKey('captions.id'), nullable=True)
    image_id = Column(Integer, ForeignKey('images.id'), nullable=True)
    vec = Column(LargeBinary)