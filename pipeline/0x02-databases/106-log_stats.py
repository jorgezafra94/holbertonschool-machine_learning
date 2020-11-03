#!/usr/bin/env python3
"""
Log stats using pymongo
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
        print('    method {}: {}'.format(elem, docs_method))

    my_filter = {"method": "GET", "path": "/status"}
    status_check = collection_logs.count_documents(my_filter)

    print('{} status check'.format(status_check))

    print('IPs:')

    for elem in collection_logs.find():
        unique_ip.append(elem['ip'])

    unique_ip = list(set(unique_ip))

    for elem in unique_ip:
        num_ip = collection_logs.count_documents({'ip': elem})
        ip_count.append((elem, num_ip))

    result = sorted(ip_count, key=lambda x: x[1], reverse=True)
    limit = 10
    if len(result) < limit:
        limit = len(result)
    result = result[:limit]
    result[0], result[1] = result[1], result[0]

    for elem in result:
        print('    {}: {}'.format(elem[0], elem[1]))
