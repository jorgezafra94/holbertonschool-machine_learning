#!/usr/bin/env python3
"""
getting all documents from a collection of mongodb
"""


def list_all(mongo_collection):
    """
    mongo_collection: input collection
    Return: list with documents of collection
    """

    documents = []

    list_all = mongo_collection.find()

    for elem in list_all:
        documents.append(elem)

    return documents
