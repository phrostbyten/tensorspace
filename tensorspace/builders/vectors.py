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
from tensorspace.util import log, get_vectors_home

BATCH_SIZE = 283

session = get_config_single_host(query_only=False)

def emit_examples(batch_size):
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

def CaptionVectors():
    log('Loading sentence embedding')
    with tf.Graph().as_default():
        sent_input = tf.placeholder(tf.string, shape=[None], name='sent_input')
        str2vec = hub.Module('https://tfhub.dev/google/universal-sentence-encoder/1')
        str2vec = str2vec(sent_input)
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            log(f'Embedding captions into vector space (batch size: {BATCH_SIZE})')
            for caps, capids in tqdm(emit_examples(BATCH_SIZE), unit=' tensor blocks'):
                vecs = sess.run(str2vec, feed_dict={'sent_input:0': caps})
                for capid, vec in zip(capids, vecs):
                    row = Vector(vec=vec, caption_id=capid)
                    session.add(row)
                    session.flush()
    session.commit()

if __name__ == '__main__':
    CaptionVectors()
  