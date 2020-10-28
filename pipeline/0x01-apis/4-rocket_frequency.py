#!/usr/bin/env python3
"""
Consuming API SpaceX
"""

import requests

if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4'
    data = requests.get(url + '/launches').json()

    my_rockets = {}

    all_rockets = []
    for elem in data:
        id_rocket = elem['rocket']
        rocket_name = requests.get(url + '/rockets/{}'.format(id_rocket))
        rocket_name = rocket_name.json()['name']
        all_rockets.append(rocket_name)

    for elem in all_rockets:
        if elem not in my_rockets.keys():
            my_rockets[elem] = 1
        else:
            my_rockets[elem] += 1

    keys = sorted(my_rockets.keys(), key=lambda x: x)
    keys = sorted(keys, key=lambda x: my_rockets[x], reverse=True)

    for key in keys:
        print('{}: {}'.format(key, my_rockets[key]))
