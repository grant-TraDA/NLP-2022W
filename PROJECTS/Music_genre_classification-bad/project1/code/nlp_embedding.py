from abc import ABC, abstractmethod
from johnsnowlabs import nlu
import numpy as np

class NLPEmbedding(ABC):
    def __init__(self):
        self.model = None
        self.name = None

    def embed_lyrics(self, data):
        pass


class GloVe(NLPEmbedding):
    def __init__(self, max_words):
        self.model = nlu.load('glove')
        self.name = 'glove'
        self.max_words = max_words
        self.embedding_size = 100

    def embed_lyrics(self, data):
        embeddings = self.model.predict(data, output_level='document')
        return np.array([self.__process_embedding(embedding) for embedding in embeddings.word_embedding_glove])
    
    def __process_embedding(self, embedding):
        embedding.resize((self.max_words, self.embedding_size), refcheck=False)
        return embedding.flatten()