#!/usr/bin/env python3
"""
TF-IDF method
this works with frequency of the words
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    * sentences is a list of sentences to analyze
    * vocab is a list of the vocabulary words to use for the analysis
    * If None, all words within sentences should be used

    Returns: embeddings, features
    * embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
    * * s is the number of sentences in sentences
    * * f is the number of features analyzed
    * features is a list of the features used for embeddings
    """
    tf_idf = TfidfVectorizer(vocabulary=vocab)
    X = tf_idf.fit_transform(sentences)
    features = tf_idf.get_feature_names()
    embeddings = X.toarray()
    return (embeddings, features)
