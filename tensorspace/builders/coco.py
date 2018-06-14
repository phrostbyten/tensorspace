import codecs
import json
import logging
import sys
from io import BytesIO
from urllib.request import urlopen, urlretrieve
from urllib.error import HTTPError
from zipfile import ZipFile

from dateutil import parser
from sqlalchemy import (Boolean, Column, Date, Float, ForeignKey, Integer,
                        SmallInteger, String, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from tqdm import tqdm

from tensorspace.model import (Caption, Category, ControlledImport, Dataset,
                              Image, License, Object, ImageFileType)
from tensorspace.util import log


utf = codecs.getreader("utf-8")
ANNOTATIONS_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'


class CocoImport(ControlledImport):
    def __init__(self, session):
        super(CocoImport, self).__init__(session, has_files=True)

    def create_database(self):
        log(f'Getting COCO annotations from {ANNOTATIONS_URL}')
        zip = ZipFile(BytesIO(urlopen(ANNOTATIONS_URL).read()))
        session = self.session

        dsinfo = Dataset(
            name='COCO',
            desc='COCO is a large-scale object detection, segmentation, and captioning dataset.',
            url='http://cocodataset.org/')
        session.add(dsinfo)
        session.commit()
        cocoid = dsinfo.id
        
        surrigate = {}

        def populate_images(index, is_training=False):
            for image in tqdm(
                    index['images'], desc='Image metadata', unit=' rows'):
                row = Image(
                    dataset_id=cocoid,
                    filetype=ImageFileType.JPEG,
                    src_filename=image['file_name'],
                    url=image['coco_url'],
                    license_id=image['license'],
                    training_split=is_training,
                    captured=parser.parse(
                        image['date_captured']),
                    orig_url=image['flickr_url'])
                session.add(row)
                session.commit()
                surrigate[image['id']] = row.id 

        def populate_annotations(index):
            for i, annotation in enumerate(
                    tqdm(index['annotations'], desc='Annotations', unit=' rows')):
                if annotation.get('bbox'):
                    row = Object(
                        image_id=surrigate[annotation['image_id']],
                        category_id=annotation['category_id'],
                        x1=annotation['bbox'][0],
                        y1=annotation['bbox'][1],
                        x2=annotation['bbox'][2],
                        y2=annotation['bbox'][3])
                else:
                    row = Caption(
                        image_id=surrigate[annotation['image_id']],
                        caption=annotation['caption'])
                session.add(row)
                if (i % 500) == 0:
                    session.flush()

        def populate_metadata(index):
            for category in index['categories']:
                row = Category(**category)
                session.add(row)
            for lic in index['licenses']:
                row = License(**lic)
                session.add(row)
            session.flush()

        with zip.open('annotations/instances_train2017.json') as fd:
            log('Processing training instances')
            index = json.load(utf(fd))
            populate_images(index, is_training=True)
            populate_annotations(index)
            populate_metadata(index)
        with zip.open('annotations/captions_train2017.json') as fd:
            log('Processing training captions')
            index = json.load(utf(fd))
            populate_annotations(index)
        with zip.open('annotations/instances_val2017.json') as fd:
            log('Processing validation instances')
            index = json.load(utf(fd))
            populate_images(index)
            populate_annotations(index)
        with zip.open('annotations/captions_val2017.json') as fd:
            log('Processing validation captions')
            index = json.load(utf(fd))
            populate_annotations(index)

        session.commit()

    def create_files(self):
        log('Downloading images')
        for url, imgid in tqdm(self.session.query(
                Image.url, Image.id), desc='Images downloaded', unit=' imgs'):
            filename = str(imgid) + '.jpg'
            if not self.exists(filename):
                try:
                    urlretrieve(url, self.path(filename))
                except HTTPError as e:
                    raise RuntimeError(f'Could not download image {url} (HTTP Code: {e.code})')



    def create(self):
        res = self.session.query(Dataset.name).filter(Dataset.name == 'COCO').first()
        if not res:
            self.create_database()
        self.create_files()

def Coco(session):
    CocoImport(session).create()

if __name__ == '__main__':
    from tensorspace.model import get_config_single_host
    Coco(get_config_single_host(initialize=True))