#!/usr/bin/env python3
"""
https://tfhub.dev/see--/bert-uncased-tf2-qa/1
loop user, bot with answer
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    * question is a string containing the question to answer
    * reference is a string containing the reference document
      from which to find the answer

    Returns: a string containing the answer
    * If no answer is found, return None
    * Your function should use the bert-uncased-tf2-qa model
      from the tensorflow-hub library
    * Your function should use the pre-trained BertTokenizer,
      bert-large-uncased-whole-word-masking-finetuned-squad,
      from the transformers library
    """
    param = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(param)
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    question_token = tokenizer.tokenize(question)
    reference_token = tokenizer.tokenize(reference)

    sep1 = ['[CLS]']
    sep2 = ['[SEP]']
    tokens = sep1 + question_token + sep2 + reference_token + sep2

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_word_ids)

    input_type_ids = [0] * (1 + len(question_token) + 1) +\
                     [1] * (len(reference_token) + 1)

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_word_ids,
                                                      input_mask,
                                                      input_type_ids))

    outputs = model([input_word_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    if answer == '':
        answer = 'Sorry, I do not understand your question.'
    return answer


def answer_loop(reference):
    """
    * reference is the reference text
    * If the answer cannot be found in the reference text,
      respond with Sorry, I do not understand your question.
    """
    # loop
    EXIT = ['exit', 'quit', 'goodbye', 'bye']

    while(1):
        user_question = input("Q: ")

        if user_question.lower() in EXIT:
            print("A: Goodbye")
            break

        else:
            print("A: {}".format(question_answer(user_question,
                                                 reference)))

    exit(0)
