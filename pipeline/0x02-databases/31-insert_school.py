#!/usr/bin/env python3
"""
insert a document in a collection using some params
"""


def insert_school(mongo_collection, **kwargs):
    """
    insert in mongo collection a new documnet
    mongo_collection: collection to insert document
    kwargs: parameters
    Return the new id created
    """

    mongo_collection.insert(kwargs)
    new_elem = mongo_collection.find(kwargs)
    return new_elem.__dict__['_Cursor__spec']['_id']
