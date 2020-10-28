#!/usr/bin/env python3
"""
Consuming API github
"""
import requests
import sys
import time

if __name__ == '__main__':
    url = sys.argv[1]
    headers = {'Accept': 'application/vnd.github.v3+json'}
    data = requests.get(url, headers=headers)

    if data.status_code == 200:
        print(data.json()['location'])

    if data.status_code == 404:
        print("Not found")

    if data.status_code == 403:
        limit_time = int(data.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        result = int((limit_time - now) / 60)
        print("Reset in {} min".format(result))
