from tensorspace.util import log, get_vectors_home
from tensorspace.model import get_config_single_host, Vector

from annoy import AnnoyIndex

class Embedding():
    def __init__(self):
        self.query = get_config_single_host()
    
    def create(self):
        log('Loading vector space')
        index = AnnoyIndex(512)
        log('Loading sentence embedding')
        for vec_id, vec in self.query(Vector.id, Vector.vec):
            index.add_item(vec_id, vec)
        index.build(10)
        index.save(str(get_vectors_home() / 'index.vec'))