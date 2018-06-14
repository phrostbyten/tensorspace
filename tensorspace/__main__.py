from tensorspace.builders import Coco, CaptionVectors
from tensorspace.model import get_config_single_host
from argparse import ArgumentParser

def do_up(action):
    session = get_config_single_host(initialize=True)
    Coco(session)
    CaptionVectors(session)

def main():
    parser = ArgumentParser()
    parser.add_argument('action', help='The action to take. Currently only "up" is supported')
    parse = parser.parse_args()
    if parse.action == 'up':
        do_up(parse.action)
    else:
        parser.error(f'Unsupported action "{parse.action}".')

if __name__ == '__main__':
    main()