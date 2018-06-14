import tensorflow as tf
import tensorflow_hub as hub
import skimage.io as io
from tqdm import tqdm
import os
from annoy import AnnoyIndex
from pathlib import Path
import logging
import sys

from tensorspace.model import get_config_single_host, Caption, Vector
from tensorspace.util import log, get_vectors_home, get_data_home

import tensorflow as tf

CAPTION_BATCH_SIZE = 283

IMG_BATCH_SIZE = 32
IMG_EPOCHS = 20

def emit_captions(session, batch_size):
    query = session.query
    caps = []
    ids = []
    res = query(Caption.id, Caption.caption)
    for image_id, caption in res:
        caps.append(caption)
        ids.append(image_id)
        if len(caps) == batch_size:
            yield caps, ids
            caps = []
            ids = []
    if len(ids) > 0:
        yield caps, ids


def get_image_module(path, batch_size, epochs, width, height):
    files = tf.data.Dataset.list_files(str(path / '*'))
    
    def decode_and_resize(fd):
        out = tf.read_file(fd)
        # In TensorFlow, "decode_jpeg" can decode PNG files too.
        # "decode_image" will not work at all, because it doesn't return the 
        # tensor's shape.
        out = tf.image.decode_jpeg(out, channels=3)
        out = tf.image.resize_images(out, [height, width], align_corners=True)
        return out

    images = files.map(lambda x: decode_and_resize(x))
    images = images.batch(IMG_BATCH_SIZE)
    images = images.repeat(IMG_EPOCHS)
    
    files = files.batch(IMG_BATCH_SIZE)
    files = files.repeat(IMG_EPOCHS)

    return tf.data.Dataset.zip((files, images))

def CaptionVectors(session):
    log('Loading sentence embedding')
    with tf.Graph().as_default():
        sent_input = tf.placeholder(tf.string, shape=[None], name='sent_input')
        str2vec = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/1')
        str2vec = str2vec(sent_input)
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            log(f'Embedding captions into vector space (batch size: {CAPTION_BATCH_SIZE})')
            for caps, capids in tqdm(emit_captions(session, CAPTION_BATCH_SIZE), unit=' tensor blocks'):
                vecs = sess.run(str2vec, feed_dict={'sent_input:0': caps})
                for capid, vec in zip(capids, vecs):
                    row = Vector(vec=vec.tobytes(), caption_id=capid)
                    session.add(row)
                session.flush()
    session.commit()


def ImageVectors(session):
    log('Loading image embedding')
    with tf.Graph().as_default():
        img2vec = hub.Module('https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/1')
        width, height = hub.get_expected_image_size(img2vec)
        dataset = get_image_module(get_data_home() / 'images', IMG_BATCH_SIZE, 1, width, height)
        iterator = dataset.make_one_shot_iterator()
        next_example = iterator.get_next()
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            try:
                while True:
                    files, images = next_example
                    vec = sess.run(img2vec(images))
                    files = sess.run(files)
                    for i, fname in enumerate(files):
                        vecid = os.path.basename(str(fname))
                        vecid = int(os.path.splitext(vecid)[0])
                        row = Vector(vec=vec[i].tobytes(), image_id=vecid)
                        session.add(row)
                    session.flush()
            except tf.errors.OutOfRangeError:
                pass # End of dataset
    session.commit()

if __name__ == '__main__':
    from tensorspace.model import get_config_single_host
    session = get_config_single_host(initialize=True)
    CaptionVectors(session)
    
  