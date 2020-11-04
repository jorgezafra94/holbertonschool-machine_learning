#!/usr/bin/env python3
"""
Log stats using pymongo
https://docs.mongodb.com/manual/core/aggregation-pipeline-optimization/
"""

from pymongo import MongoClient


if __name__ == "__main__":
    """
    again printing some information from logs mongo db
    """

    unique_ip = []
    ip_count = []
    
    client = MongoClient('mongodb://127.0.0.1:27017')

    collection_logs = client.logs.nginx

    num_logs = collection_logs.count_documents({})
    
    print('{} logs'.format(num_logs))

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    print('Methods:')
    
    for elem in methods:
        docs_method = collection_logs.count_documents({'method': elem})
        print("\tmethod {}: {}".format(elem, docs_method))

    my_query= {"method": "GET", "path": "/status"}
    status_check = collection_logs.count_documents(my_query)

    print('{} status check'.format(status_check))

    print('IPs:')

    pipeline = [
        {"$sortByCount": '$ip'},
        {"$limit": 10},
        {"$sort": {"ip": -1}},

    ]
    ips = collection.aggregate(pipeline)

    for ip in ips:
        print("\t{}: {}".format(ip['_id'], ip['count']))
