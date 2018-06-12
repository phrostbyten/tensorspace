from tensorspace.builders import Coco, CaptionVectors
from argparse import ArgumentParser

def do_up(action):
    Coco()
    CaptionVectors()

parser = ArgumentParser()
parser.add_argument('action', help='The action to take. Currently only "up" is supported')
parse = parser.parse_args()
if parse.action == 'up':
    do_up(parse.action)
else:
    parser.error(f'Unsupported action "{parse.action}".')