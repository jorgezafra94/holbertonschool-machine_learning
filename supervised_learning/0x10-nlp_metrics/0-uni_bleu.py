#!/usr/bin/env python3
"""
Precision BLEU
Bilingual evaluation understudy
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    * references is a list of reference translations
    * each reference translation is a list of the words in the translation
    * sentence is a list containing the model proposed sentence

    Returns: the unigram BLEU score
    """

    uniques = list(set(sentence))
    dict_words = {}

    for reference in references:
        for word in reference:
            if word in uniques:
                if word not in dict_words.keys():
                    dict_words[word] = reference.count(word)
                else:
                    actual = reference.count(word)
                    prev = dict_words[word]
                    dict_words[word] = max(actual, prev)

    len_cand = len(sentence)
    prob = sum(dict_words.values()) / len_cand

    best_match_tuples = []
    for reference in references:
        ref_len = len(reference)
        diff = abs(ref_len - len_cand)
        best_match_tuples.append((diff, ref_len))

    sort_tuples = sorted(best_match_tuples, key=lambda x: x[0])
    best_match = sort_tuples[0][1]

    # Brevity penalty
    if len_cand > best_match:
        bp = 1
    else:
        bp = np.exp(1 - (best_match / len_cand))

    Bleu_score = bp * np.exp(np.log(prob))
    return Bleu_score
