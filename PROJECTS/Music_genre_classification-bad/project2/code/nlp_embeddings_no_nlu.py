from abc import ABC
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer


class NLPEmbeddingTorch(ABC):
    '''Abstract class of an embedding method based on torch.'''
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


class DistilBERT(nn.Module):
    '''DistilBERT embedding method.'''
    def __init__(self, max_words, device):
        super().__init__()
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.name = 'DistilBERT'
        self.max_words = max_words
        self.embedding_size = self.model.config.dim
        self.device = device

    def embed_lyrics(self, data):
        embeddings = np.empty((data.shape[0], self.embedding_size))
        for i, lyric in enumerate(data):
            encoding = self.tokenizer.encode_plus(
                lyric,
                add_special_tokens=True,
                max_length=self.max_words,
                truncation=True,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            with torch.no_grad():
                model_output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask)
            embeddings[i] = model_output.last_hidden_state.mean(axis=1)[0]
        return embeddings


class SentenceTransformerMPNET(nn.Module):
    '''SentenceTransformerMPNET embedding method.'''
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.name = 'SentenceTransformerMPNET'
        self.embedding_size = self.model.get_sentence_embedding_dimension()

    def embed_lyrics(self, data):
        embeddings = np.empty((data.shape[0], self.embedding_size))
        for i, lyric in enumerate(data):
            embeddings[i] = self.model.encode(lyric)
        return embeddings
