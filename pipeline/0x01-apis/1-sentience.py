#!/usr/bin/env python3
"""
home planets of all sentient species.
star wars
"""

import requests


def sentientPlanets():
    """
    getting all sentient planets
    """

    data = requests.get('https://swapi-api.hbtn.io/api/species/').json()

    sentient_planets = []

    while (data['next']):
        for r in data['results']:
            if (r['designation'] == 'sentient'
                    or r['classification'] == 'sentient'):
                if r['homeworld']:
                    planet = requests.get(r['homeworld']).json()
                    sentient_planets.append(planet['name'])

        data = requests.get(data['next'])
        data = data.json()

    if data['next'] is None:
        for r in data['results']:
            if (r['designation'] == 'sentient'
                    or r['classification'] == 'sentient'):
                if r['homeworld']:
                    planet = requests.get(r['homeworld']).json()
                    sentient_planets.append(planet['name'])

    return sentient_planets
