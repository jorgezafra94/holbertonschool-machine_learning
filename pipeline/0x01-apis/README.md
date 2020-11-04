# API's
## Task0 - Can I join?
By using the [Swapi API](https://swapi-api.hbtn.io/), create a method that returns the list of ships that can hold a given number of passengers:<br>
<br>
* Prototype: def availableShips(passengerCount):
* Don’t forget the pagination
* If no ship available, return an empty list.

```
ubuntu:~$ cat 0-main.py
#!/usr/bin/env python3
"""
Test file
"""
availableShips = __import__('0-passengers').availableShips
ships = availableShips(4)
for ship in ships:
    print(ship)

ubuntu:~$ ./0-main.py
CR90 corvette
Sentinel-class landing craft
Death Star
Millennium Falcon
Executor
Rebel transport
Slave 1
Imperial shuttle
EF76 Nebulon-B escort frigate
Calamari Cruiser
Republic Cruiser
Droid control ship
Scimitar
J-type diplomatic barge
AA-9 Coruscant freighter
Republic Assault ship
Solar Sailer
Trade Federation cruiser
Theta-class T-2c shuttle
Republic attack cruiser
ubuntu:~$
```

## Task1 - Where I am?
By using the  [Swapi API](https://swapi-api.hbtn.io/), create a method that returns the list of names of the home planets of all sentient species.<br>
<br>
* Prototype: def sentientPlanets():
* Don’t forget the pagination

```
ubuntu:~$ cat 1-main.py
#!/usr/bin/env python3
"""
Test file
"""
sentientPlanets = __import__('1-sentience').sentientPlanets
planets = sentientPlanets()
for planet in planets:
    print(planet)

ubuntu:~$ ./1-main.py
Endor
Naboo
Coruscant
Kamino
Geonosis
Utapau
Kashyyyk
Cato Neimoidia
Rodia
Nal Hutta
unknown
Trandosha
Mon Cala
Sullust
Toydaria
Malastare
Ryloth
Aleen Minor
Vulpter
Troiken
Tund
Cerea
Glee Anselm
Iridonia
Tholoth
Iktotch
Quermia
Dorin
Champala
Mirial
Zolan
Ojom
Skako
Muunilinst
Shili
Kalee
ubuntu:~$
```

## Task2 - Rate me is you can!
By using the [Github API](https://docs.github.com/en/free-pro-team@latest/rest/reference/users), write a script that prints the location of a specific user:<br>
<br>
* The user is passed as first argument of the script with the full API URL, example: ./2-user_location.py https://api.github.com/users/holbertonschool
* If the user doesn’t exist, print Not found
* If the status code is 403, print Reset in X min where X is the number of minutes from now and the value of X-Ratelimit-Reset
* Your code should not be executed when the file is imported (you should use if __name__ == '__main__':)

```
ubuntu:~$ ./2-user_location.py https://api.github.com/users/holbertonschool
San Francisco, CA
ubuntu:~$
ubuntu:~$ ./2-user_location.py https://api.github.com/users/holberton_ho_no
Not found
ubuntu:~$
... after a lot of request... 60 exactly...
ubuntu:~$
ubuntu:~$ ./2-user_location.py https://api.github.com/users/holbertonschool
Reset in 16 min
ubuntu:~$ 
```

## Task3 - What will be next?
By using the (unofficial) [SpaceX API](https://github.com/r-spacex/SpaceX-API/blob/master/docs/v4/README.md), write a script that displays the upcoming launch with these information:<br>
<br>
* Name of the launch
* The date (in local time)
* The rocket name
* The name (with the locality) of the launchpad
* Format: <launch name> (<date>) <rocket name> - <launchpad name> (<launchpad locality>)

* The “upcoming launch” is the one which is the soonest from now, in UTC (we encourage you to use the date_unix for sorting it) - and if 2 launches have the same date, use the first one in the API result.
* Your code should not be executed when the file is imported (you should use if __name__ == '__main__':)


```
ubuntu:~$ ./3-upcoming.py 
Starlink-12 (v1.0) (2020-09-27T10:43:00-04:00) Falcon 9 - KSC LC 39A (Cape Canaveral)
ubuntu:~$
```

## Task4 -  How many by rocket? 
By using the (unofficial) [SpaceX API](https://github.com/r-spacex/SpaceX-API/blob/master/docs/v4/README.md), write a script that displays the number of launches per rocket.<br>
<br>
* All launches should be taking in consideration
* Each line should contain the rocket name and the number of launches separated by : (format below in the example)
* Order the result by the number launches (descending)
* If multiple rockets have the same amount of launches, order them by alphabetic order (A to Z)
* Your code should not be executed when the file is imported (you should use if __name__ == '__main__':)

```
ubuntu:~$ ./4-rocket_frequency.py
Falcon 9: 105
Falcon 1: 5
Falcon Heavy: 3
ubuntu:~$ 
```
