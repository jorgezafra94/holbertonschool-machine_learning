#!/usr/bin/env python3
"""
Log stats using pymongo
"""

from pymongo import MongoClient


if __name__ == "__main__":
    """
    * Database: logs
    * Collection: nginx
    * Display (same as the example):
      - first line: x logs where x is the number of
        documents in this collection
      - second line: Methods:
          - 5 lines with the number of documents with the
            method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
            in this order (see example below - warning: it’s a
            tabulation before each line)
          - one line with the number of documents with:
            method=GET
            path=/status
    """

    client = MongoClient('mongodb://127.0.0.1:27017')

    collection_logs = client.logs.nginx

    num_docs = collection_logs.count_documents({})

    print("{} logs".format(num_docs))

    print("Methods:")

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        num_met = collection_logs.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, num_met))

    my_dicti = {"method": "GET", "path": "/status"}

    num_dicti = collection_logs.count_documents(my_dicti)
    print("{} status check".format(num_dicti))
