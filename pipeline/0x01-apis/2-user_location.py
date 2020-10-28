#!/usr/bin/env python3
"""
rate github api
"""

import sys
import requests
import time

if __name__ == '__main__':
    header = {'Accept': 'application/vnd.github.v3+json'}
    url = sys.argv[1]

    data = requests.get(url, params=header)

    if data.status_code == 200:
        response = data.json()
        print(response['location'])

    elif data.status_code == 404:
        print('Not found')

    elif data.status_code == 403:
        ratelimit = data.headers['X-Ratelimit-Reset']
        now = time.time()
        result = int((int(ratelimit) - int(now)) / 60)
        print('Reset in {} min'.format(result))
