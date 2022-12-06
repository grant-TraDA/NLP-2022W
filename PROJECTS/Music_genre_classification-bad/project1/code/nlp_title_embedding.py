from abc import ABC, abstractmethod
from johnsnowlabs import nlu
from nlp_classifier import NLPClassifier
import numpy as np

class NLPTitleEmbedding(ABC):
    def __init__(self, nlp_classifier):
        self.model = None
        self.name = None
        self.nlp_classifier = nlp_classifier
    
    def get_title_lyrics_embedding(self, lyrics, title):
        pass

class GloVeTitle(NLPTitleEmbedding):
    def __init__(self, max_words, max_words_title):
        self.model = nlu.load('glove')
        self.name = 'gloveT'
        self.max_words = max_words
        self.max_words_title = max_words_title
        self.embedding_size = 100
    
    def get_title_lyrics_embedding(self, lyrics, title):
        title = self.__embed(title, self.max_words_title)
        lyrics = self.__embed(lyrics, self.max_words)
        return np.concatenate((title, lyrics), axis=1)

    def __embed(self, data, max_words):
        embeddings = self.model.predict(data, output_level='document')
        return np.array([self.__process_embedding(embedding, max_words) for embedding in embeddings.word_embedding_glove])
    
    def __process_embedding(self, embedding, max_words):
        embedding.resize((max_words, self.embedding_size), refcheck=False)
        return embedding.flatten()