#!/usr/bin/env python3
""" script that prints the location of a specific github user"""

import sys
import requests
import time

if __name__ == '__main__':
    if len(sys.argv) > 1:
        header = {'Accept': 'application/vnd.github.v3+json'}
        url = sys.argv[1]

        data = requests.get(url, params=header)

        if data.status_code == 403:
            reset = data.headers['X-Ratelimit-Reset']
            X = int((int(reset) - int(time.time())) / 60)
            print("Reset in {} min".format(X))

        elif data.status_code == 200:
            response = data.json()
            print(response['location'])

        elif data.status_code == 404:
            print('Not found')
