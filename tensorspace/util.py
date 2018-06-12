import logging
import sys 
from pathlib import Path
import os

PROGRESS = 25

logging.basicConfig(stream=sys.stdout, level=PROGRESS)
logging.addLevelName(PROGRESS, "PROGRESS")
pb_logger = logging.getLogger('tensorspace')

def log(msg):
    pb_logger.log(PROGRESS, msg)

def get_data_home():
    home = os.environ.get('PBHOME') or Path.home() / '.tensorspace'
    home = home / 'data'
    if not home.is_dir():
        Path.mkdir(home, parents=True)
    return home

def get_vectors_home():
    home = os.environ.get('PBHOME') or Path.home() / '.tensorspace'
    home = home / 'vectors'
    if not home.is_dir():
        Path.mkdir(home, parents=True)
    return home