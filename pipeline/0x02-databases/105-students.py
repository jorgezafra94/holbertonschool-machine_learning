#!/usr/bin/env python3
"""
function that returns all students sorted by average score
"""


def top_students(mongo_collection):
    """
    * mongo_collection will be the pymongo collection object
    * The top must be ordered
    * The average score must be part of each item returns
      with key = averageScore
    """

    all_list = mongo_collection.find()
    students = []

    for document in all_list:
        value = 0
        count = 0

        for topic in document['topics']:
            value += topic['score']
            count += 1

        value = value / count

        document['averageScore'] = value

        students.append(document)

    result = sorted(students, key=lambda x: x["averageScore"], reverse=True)

    return result
