#!/usr/bin/env python3
"""
Consuming API SpaceX
"""

import requests

if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4'
    data = requests.get(url + '/launches/upcoming').json()

    date_launch = [elem['date_unix'] for elem in data]

    index = date_launch.index(min(date_launch))

    launch = data[index]

    id_launch = launch['id']
    id_rocket = launch['rocket']
    id_launch_pad = launch['launchpad']

    launch_name = requests.get(url + '/launches/{}'.format(id_launch))
    launch_name = launch_name.json()['name']
    date_local = launch['date_local']
    rocket_name = requests.get(url + '/rockets/{}'.format(id_rocket))
    rocket_name = rocket_name.json()['name']
    launchpad = requests.get(url + '/launchpads/{}'.format(id_launch_pad))
    launchpad = launchpad.json()['name']
    launchpad_loc = requests.get(url + '/launchpads/{}'.format(id_launch_pad))
    launchpad_loc = launchpad_loc.json()['locality']

    result = ''
    result += '{}'.format(launch_name)
    result += ' ({})'.format(date_local)
    result += ' {}'.format(rocket_name)
    result += ' - {}'.format(launchpad)
    result += ' ({})'.format(launchpad_loc)

    print(result)
