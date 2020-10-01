#!/usr/bin/env python3
"""
based on:
https://www.tensorflow.org/tutorials/text/transformer
Creating Class Dataset
"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """
    Class Dataset
    """

    def __init__(self):
        """
        * data_train, which contains the ted_hrlr_translate/pt_to_en
          tf.data.Dataset train split, loaded as_supervided
        * data_valid, which contains the ted_hrlr_translate/pt_to_en
          tf.data.Dataset validate split, loaded as_supervided
        * tokenizer_pt is the Portuguese tokenizer created from the
          training set
        * tokenizer_en is the English tokenizer created from the
          training set
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']

        PT, EN = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = PT, EN

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        * data is a tf.data.Dataset whose examples are formatted
          as a tuple (pt, en)
        * * pt is the tf.Tensor containing the Portuguese sentence
        * * en is the tf.Tensor containing the corresponding English
            sentence
        * The maximum vocab size should be set to 2**15

        Returns: tokenizer_pt, tokenizer_en
        * tokenizer_pt is the Portuguese tokenizer
        * tokenizer_en is the English tokenizer
        """
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2 ** 15)

        return (tokenizer_pt, tokenizer_en)

    def encode(self, pt, en):
        """
        * pt is the tf.Tensor containing the Portuguese sentence
        * en is the tf.Tensor containing the corresponding English
          sentence
        * The tokenized sentences should include the start and end
          of sentence tokens
        * The start token should be indexed as vocab_size
        * The end token should be indexed as vocab_size + 1

        Returns: pt_tokens, en_tokens
        * pt_tokens is a tf.Tensor containing the Portuguese tokens
        * en_tokens is a tf.Tensor containing the English tokens
        """
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return lang1, lang2

    def tf_encode(self, pt, en):
        """
        Make sure to set the shape of the pt and en return tensors
        """
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
