#!/usr/bin/env python3
"""
Precision BLEU
Bilingual evaluation understudy
"""

import numpy as np


def get_gram(sentence, n):
    """
    create grams from sentence
    """
    list_grams_cand = []
    for i in range(len(sentence)):
        last = i + n
        begin = i
        if last >= len(sentence) + 1:
            break
        aux = sentence[begin: last]
        result = ' '.join(aux)
        list_grams_cand.append(result)
    return list_grams_cand


def ngram_bleu(references, sentence, n):
    """
    * references is a list of reference translations
    * each reference translation is a list of the words in the translation
    * sentence is a list containing the model proposed sentence
    * n is the size of the n-gram to use for evaluation

    Returns: the n-gram BLEU score
    """
    dict_words = {}
    # total_words = {}
    # getting grams candidate
    cand_grams = get_gram(sentence, n)
    cand_grams = list(set(cand_grams))
    len_cand = len(cand_grams)

    # getting grams references
    reference_grams = []
    for reference in references:
        list_grams = get_gram(reference, n)
        reference_grams.append(list_grams)

    for grams in reference_grams:
        for word in grams:
            if word in cand_grams:
                if word not in dict_words.keys():
                    dict_words[word] = grams.count(word)
                else:
                    actual = grams.count(word)
                    prev = dict_words[word]
                    dict_words[word] = max(actual, prev)

    prob = sum(dict_words.values()) / len_cand

    # careful we have to do this step using the original sentence
    best_match_tuples = []
    for reference in references:
        ref_len = len(reference)
        diff = abs(ref_len - len(sentence))
        best_match_tuples.append((diff, ref_len))

    sort_tuples = sorted(best_match_tuples, key=lambda x: x[0])
    best_match = sort_tuples[0][1]

    # Brevity penalty
    if len_cand > best_match:
        bp = 1
    else:
        bp = np.exp(1 - (best_match / len(sentence)))

    Bleu_score = bp * np.exp(np.log(prob))
    return Bleu_score
