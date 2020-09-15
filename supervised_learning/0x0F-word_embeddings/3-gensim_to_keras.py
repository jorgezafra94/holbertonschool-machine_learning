#!/usr/bin/env python3
"""
Creating a word2vec keras
"""

def gensim_to_keras(model):
    """
    gensim model to keras
    """
    return model.wv.get_keras_embedding(train_embeddings=False)
