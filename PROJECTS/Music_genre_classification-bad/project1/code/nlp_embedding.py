from abc import ABC, abstractmethod
from johnsnowlabs import nlu
import numpy as np

class NLPEmbedding(ABC):
    '''Abstract class of an embedding method.'''
    def __init__(self):
        self.model = None
        self.name = None
        self.max_words = None
        self.embedding_size = None
        self.column = None

    def embed_lyrics(self, data):
        '''
        Embed lyrics of the data
        Parameters:
            data (Series): Lyrics of the data.
        Returns:
            ndarray: Embedded lyrics.
        '''
        embeddings = self.model.predict(data, output_level='document')
        return np.array([self.__process_embedding(embedding) for embedding in embeddings[self.column]])

    def __process_embedding(self, embedding):
        '''
        Resize embedding to desired size and flatten the array.
        Parameters:
            embedding (ndarray): Embedded lyrics.
        Returns:
            ndarray: Processed embeddings.
        '''
        embedding.resize((self.max_words, self.embedding_size), refcheck=False)
        return embedding.flatten()


class GloVe(NLPEmbedding):
    '''GloVe embedding method.'''
    def __init__(self, max_words):
        self.model = nlu.load('glove')
        self.name = 'glove'
        self.max_words = max_words
        self.embedding_size = 100
        self.column = 'word_embedding_glove'

class SmallBert(NLPEmbedding):
    '''Smaller BERT embedding method.'''
    def __init__(self, max_words):
        self.model = nlu.load('en.embed.bert.small_L2_128')
        self.name = 'smaller-bert'
        self.max_words = max_words
        self.embedding_size = 128
        self.column = 'word_embedding_bert'

class Bert(NLPEmbedding):
    '''BERT embedding method.'''
    def __init__(self, max_words):
        self.model = nlu.load('en.embed.bert')
        self.name = 'bert'
        self.max_words = max_words
        self.embedding_size = 768
        self.column = 'word_embedding_bert'

class LargeBert(NLPEmbedding):
    '''Large BERT embedding method.'''
    def __init__(self, max_words):
        self.model = nlu.load('en.embed.bert.large_uncased')
        self.name = 'large-bert'
        self.max_words = max_words
        self.embedding_size = 1024
        self.column = 'word_embedding_bert'

class Word2vec(NLPEmbedding):
    '''Word2vec embedding method.'''
    def __init__(self, max_words):
        self.model = nlu.load('en.embed.word2vec.gigaword')
        self.name = 'word2vec'
        self.max_words = max_words
        self.embedding_size = 300
        self.column = 'word_embedding_UNIQUE'