#!/usr/bin/env python3
"""
Getting Ships of Star Wars API
using  pagination
"""

import requests


def availableShips(passengerCount):
    """
    passangerCount: number of passanger to hold
    """
    data = requests.get('https://swapi-api.hbtn.io/api/starships/')
    data = data.json()
    my_ships = []

    # if there are pagination
    while(data['next']):
        for result in data['results']:
            capacity = result['passengers']
            capacity = capacity.replace(',', '')
            if capacity.isnumeric():
                if int(capacity) >= passengerCount:
                    my_ships.append(result['name'])
        data = requests.get(data['next'])
        data = data.json()

    # if this is  the last page
    if data['next'] is None:
        for result in data['results']:
            capacity = result['passengers']
            capacity = capacity.replace(',', '')
            if capacity.isnumeric():
                if int(capacity) >= passengerCount:
                    my_ships.append(result['name'])

    return my_ships
