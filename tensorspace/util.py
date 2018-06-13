import logging
import sys 
from pathlib import Path
import os

STATUS = 25

logging.basicConfig(stream=sys.stdout, level=STATUS)
logging.addLevelName(STATUS, "STATUS")
pb_logger = logging.getLogger('tensorspace')

def log(msg):
    pb_logger.log(STATUS, msg)

def get_data_home():
    home = os.environ.get('PBHOME') or Path.home() / '.phrostbyte'
    home = home / 'data'
    if not home.is_dir():
        Path.mkdir(home, parents=True)
    return home

def get_vectors_home():
    home = os.environ.get('PBHOME') or Path.home() / '.phrostbyte'
    home = home / 'vectors'
    if not home.is_dir():
        Path.mkdir(home, parents=True)
    return home