# DATABASES
## MYSQL
```
$  sudo apt-get install mysql-server
...
$ mysql -uroot -p
Enter password: 
Welcome to the MySQL monitor.  Commands end with ; or \g.
Your MySQL connection id is 5
Server version: 5.7.31-0ubuntu0.16.04.1 (Ubuntu)

Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

mysql>
```

## MONGODB
```
$ wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | apt-key add -
$ echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.2 multiverse" > /etc/apt/sources.list.d/mongodb-org-4.2.list
$ sudo apt-get update
$ sudo apt-get install -y mongodb-org
...
$  sudo service mongod status
mongod start/running, process 3627
$ mongo --version
MongoDB shell version v4.2.8
git version: 43d25964249164d76d5e04dd6cf38f6111e21f5f
OpenSSL version: OpenSSL 1.1.1  11 Sep 2018
allocator: tcmalloc
modules: none
build environment:
    distmod: ubuntu1804
    distarch: x86_64
    target_arch: x86_64
$  
$ pip3 install pymongo
$ python3
>>> import pymongo
>>> pymongo.__version__
'3.10.1'
```

## Task0 - Create a database
Write a script that creates the database db_0 in your MySQL server.<br>
<br>
* If the database db_0 already exists, your script should not fail
* You are not allowed to use the SELECT or SHOW statements

```
ubuntu:~/$ cat 0-create_database_if_missing.sql | mysql -hlocalhost -uroot -p
Enter password: 
ubuntu:~/$ echo "SHOW databases;" | mysql -hlocalhost -uroot -p
Enter password: 
Database
information_schema
db_0
mysql
performance_schema
ubuntu:~/$ cat 0-create_database_if_missing.sql | mysql -hlocalhost -uroot -p
Enter password: 
ubuntu:~/$
```

## Task1 -  First table
Write a script that creates a table called first_table in the current database in your MySQL server.<br>
<br>
* first_table description:
* * id INT
* * name VARCHAR(256)
* The database name will be passed as an argument of the mysql command
* If the table first_table already exists, your script should not fail
* You are not allowed to use the SELECT or SHOW statements

```
ubuntu:~/$ cat 1-first_table.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
ubuntu:~/$ echo "SHOW TABLES;" | mysql -hlocalhost -uroot -p db_0
Enter password: 
Tables_in_db_0
first_table
ubuntu:~/$
```

## Task2 - 2. List all in table
Write a script that lists all rows of the table first_table in your MySQL server.<br>
<br>
* All fields should be printed
* The database name will be passed as an argument of the mysql command

```
ubuntu:~/$ cat 2-list_values.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
ubuntu:~/$ 
```

## Task3 - First add
Write a script that inserts a new row in the table first_table in your MySQL server.<br>
<br>
* New row:
* * id = 89
* * name = Holberton School
* The database name will be passed as an argument of the mysql command
```
ubuntu:~/$ cat 3-insert_value.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
ubuntu:~/$ cat 2-list_values.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
id  name
89  Holberton School
ubuntu:~/$ cat 3-insert_value.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
ubuntu:~/$ cat 3-insert_value.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
ubuntu:~/$ cat 2-list_values.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
id  name
89  Holberton School
89  Holberton School
89  Holberton School
ubuntu:~/$ 
```

## Task4 - Select the best
Write a script that lists all records with a score >= 10 in the table second_table in your MySQL server.<br>
<br>
* Results should display both the score and the name (in this order)
* Records should be ordered by score (top first)
* The database name will be passed as an argument of the mysql command

```
ubuntu:~/$ cat setup.sql
-- Create table and insert data
CREATE TABLE IF NOT EXISTS second_table (
    id INT,
    name VARCHAR(256),
    score INT
);
INSERT INTO second_table (id, name, score) VALUES (1, "Bob", 14);
INSERT INTO second_table (id, name, score) VALUES (2, "Roy", 3);
INSERT INTO second_table (id, name, score) VALUES (3, "John", 10);
INSERT INTO second_table (id, name, score) VALUES (4, "Bryan", 8);

ubuntu:~/$ cat setup.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
ubuntu:~/$ cat 4-best_score.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
score   name
14  Bob
10  John
ubuntu:~/$ 
```

## Task5 - Average
Write a script that computes the score average of all records in the table second_table in your MySQL server.<br>
<br>
* The result column name should be average
* The database name will be passed as an argument of the mysql command

```
ubuntu:~/$ cat setup.sql
-- Create table and insert data
CREATE TABLE IF NOT EXISTS second_table (
    id INT,
    name VARCHAR(256),
    score INT
);
INSERT INTO second_table (id, name, score) VALUES (1, "Bob", 14);
INSERT INTO second_table (id, name, score) VALUES (2, "Roy", 5);
INSERT INTO second_table (id, name, score) VALUES (3, "John", 10);
INSERT INTO second_table (id, name, score) VALUES (4, "Bryan", 8);

ubuntu:~/$ cat setup.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
ubuntu:~/$ cat 5-average.sql | mysql -hlocalhost -uroot -p db_0
Enter password: 
average
9.25
ubuntu:~/$ 
```

## Task6 - Temperatures #0
Import in hbtn_0c_0 database this table dump: [download](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/272/temperatures.sql)<br>
<br>
* Write a script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending).

```
# rember always create the hbtn_0c_0 (database) in you mysql server, once you create it you can use this
# mysql -uroot -p hbtn_0c_0 < ./temperatures.sql

ubuntu:~/$ cat 6-avg_temperatures.sql | mysql -hlocalhost -uroot -p hbtn_0c_0
Enter password: 
city    avg_temp
Chandler    72.8627
Gilbert 71.8088
Pismo beach 71.5147
San Francisco   71.4804
Sedona  70.7696
Phoenix 70.5882
Oakland 70.5637
Sunnyvale   70.5245
Chicago 70.4461
San Diego   70.1373
Glendale    70.1225
Sonoma  70.0392
Yuma    69.3873
San Jose    69.2990
Tucson  69.0245
Joliet  68.6716
Naperville  68.1029
Tempe   67.0441
Peoria  66.5392
ubuntu:~/$ 
```

## Task7 - Temperatures #2 
Import in hbtn_0c_0 database<br>
<br>
* Write a script that displays the max temperature of each state (ordered by State name).

```
ubuntu:~/$ cat 7-max_state.sql | mysql -hlocalhost -uroot -p hbtn_0c_0
Enter password: 
state   max_temp
AZ  110
CA  110
IL  110
ubuntu:~/$ 
```

## Task8 - Genre ID by show
Import the database dump from hbtn_0d_tvshows to your MySQL server: [download](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows.sql)<br>
<br>
Write a script that lists all shows contained in hbtn_0d_tvshows that have at least one genre linked.<br>
* Each record should display: tv_shows.title - tv_show_genres.genre_id
* Results must be sorted in ascending order by tv_shows.title and tv_show_genres.genre_id
* You can use only one SELECT statement
* The database name will be passed as an argument of the mysql command

```
ubuntu:~/$ cat 8-genre_id_by_show.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows
Enter password: 
title   genre_id
Breaking Bad    1
Breaking Bad    6
Breaking Bad    7
Breaking Bad    8
Dexter  1
Dexter  2
Dexter  6
Dexter  7
Dexter  8
Game of Thrones 1
Game of Thrones 3
Game of Thrones 4
House   1
House   2
New Girl    5
Silicon Valley  5
The Big Bang Theory 5
The Last Man on Earth   1
The Last Man on Earth   5
ubuntu:~/$ 
```

## Task9 - No genre
Import the database dump from hbtn_0d_tvshows<br>
<br>
Write a script that lists all shows contained in hbtn_0d_tvshows without a genre linked.<br>
* Each record should display: tv_shows.title - tv_show_genres.genre_id
* Results must be sorted in ascending order by tv_shows.title and tv_show_genres.genre_id
* You can use only one SELECT statement
* The database name will be passed as an argument of the mysql command

```
ubuntu:~/$ cat 9-no_genre.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows
Enter password: 
title   genre_id
Better Call Saul    NULL
Homeland    NULL
ubuntu:~/$ 
```

## Task10 - Number of shows by genre
Import the database dump from hbtn_0d_tvshows<br>
<br>
Write a script that lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.<br>
* Each record should display: <TV Show genre> - <Number of shows linked to this genre>
* First column must be called genre
* Second column must be called number_of_shows
* Don’t display a genre that doesn’t have any shows linked
* Results must be sorted in descending order by the number of shows linked
* You can use only one SELECT statement
* The database name will be passed as an argument of the mysql command
  
```
ubuntu:~/$ cat 10-count_shows_by_genre.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows
Enter password: 
genre   number_of_shows
Drama   5
Comedy  4
Mystery 2
Crime   2
Suspense    2
Thriller    2
Adventure   1
Fantasy 1
ubuntu:~/$ 
```

## Task11 - Rotten tomatoes
Import the database hbtn_0d_tvshows_rate dump to your MySQL server: [download](https://s3.amazonaws.com/intranet-projects-files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows_rate.sql)<br>
<br>
Write a script that lists all shows from hbtn_0d_tvshows_rate by their rating.<br>
* Each record should display: tv_shows.title - rating sum
* Results must be sorted in descending order by the rating
* You can use only one SELECT statement
* The database name will be passed as an argument of the mysql command

```
ubuntu:~/$ cat 11-rating_shows.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows_rate
Enter password: 
title   rating
Better Call Saul    163
Homeland    145
Silicon Valley  82
Game of Thrones 79
Dexter  24
House   21
Breaking Bad    16
The Last Man on Earth   10
The Big Bang Theory 0
New Girl    0
ubuntu:~/$ 
```

## Task12 - Best genre
Import the database dump from hbtn_0d_tvshows_rate<br>
<br>
Write a script that lists all genres in the database hbtn_0d_tvshows_rate by their rating.<br>
* Each record should display: tv_genres.name - rating sum
* Results must be sorted in descending order by their rating
* You can use only one SELECT statement
* The database name will be passed as an argument of the mysql command

```
ubuntu:~/$ cat 12-rating_genres.sql | mysql -hlocalhost -uroot -p hbtn_0d_tvshows_rate
Enter password: 
name    rating
Drama   150
Comedy  92
Adventure   79
Fantasy 79
Mystery 45
Crime   40
Suspense    40
Thriller    40
ubuntu:~/$ 
```

## Task13 - We are all unique!
Write a SQL script that creates a table users following these requirements:<br>
<br>
* With these attributes:
* * id, integer, never null, auto increment and primary key
* * email, string (255 characters), never null and unique
* * name, string (255 characters)
* If the table already exists, your script should not fail
* Your script can be executed on any database

```
ubuntu:~$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
Enter password: 
ERROR 1146 (42S02) at line 1: Table 'holberton.users' doesn't exist
ubuntu:~$ 
ubuntu:~$ cat 13-uniq_users.sql | mysql -uroot -p holberton
Enter password: 
ubuntu:~$ 
ubuntu:~$ echo 'INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Bob");' | mysql -uroot -p holberton
Enter password: 
ubuntu:~$ echo 'INSERT INTO users (email, name) VALUES ("sylvie@dylan.com", "Sylvie");' | mysql -uroot -p holberton
Enter password: 
ubuntu:~$ echo 'INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Jean");' | mysql -uroot -p holberton
Enter password: 
ERROR 1062 (23000) at line 1: Duplicate entry 'bob@dylan.com' for key 'email'
ubuntu:~$ 
ubuntu:~$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
Enter password: 
id  email   name
1   bob@dylan.com   Bob
2   sylvie@dylan.com    Sylvie
ubuntu:~$ 
```

## Task14 - In and not out
Write a SQL script that creates a table users following these requirements:<br>
<br>
* With these attributes:
* * id, integer, never null, auto increment and primary key
* * email, string (255 characters), never null and unique
* * name, string (255 characters)
* * country, enumeration of countries: US, CO and TN, never null (= default will be the first element of the enumeration, here US)
* If the table already exists, your script should not fail
* Your script can be executed on any database

```
ubuntu:~$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
Enter password: 
ERROR 1146 (42S02) at line 1: Table 'holberton.users' doesn't exist
ubuntu:~$ 
ubuntu:~$ cat 14-country_users.sql | mysql -uroot -p holberton
Enter password: 
ubuntu:~$ 
ubuntu:~$ echo 'INSERT INTO users (email, name, country) VALUES ("bob@dylan.com", "Bob", "US");' | mysql -uroot -p holberton
Enter password: 
ubuntu:~$ echo 'INSERT INTO users (email, name, country) VALUES ("sylvie@dylan.com", "Sylvie", "CO");' | mysql -uroot -p holberton
Enter password: 
ubuntu:~$ echo 'INSERT INTO users (email, name, country) VALUES ("jean@dylan.com", "Jean", "FR");' | mysql -uroot -p holberton
Enter password: 
ERROR 1265 (01000) at line 1: Data truncated for column 'country' at row 1
ubuntu:~$ 
ubuntu:~$ echo 'INSERT INTO users (email, name) VALUES ("john@dylan.com", "John");' | mysql -uroot -p holberton
Enter password: 
ubuntu:~$ 
ubuntu:~$ echo "SELECT * FROM users;" | mysql -uroot -p holberton
Enter password: 
id  email   name    country
1   bob@dylan.com   Bob US
2   sylvie@dylan.com    Sylvie  CO
3   john@dylan.com  John    US
ubuntu:~$ 
```

## Task15 - Best band ever!
Write a SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans<br>
<br>
Requirements:
* Import this table dump: [metal_bands.sql.zip](https://intranet.hbtn.io/rltoken/O5SYbtdVdsalsjWWA_v3-g)
* Column names must be: origin and nb_fans
* Your script can be executed on any database

```
ubuntu:~$ cat metal_bands.sql | mysql -uroot -p holberton
Enter password: 
ubuntu:~$ 
ubuntu:~$ cat 15-fans.sql | mysql -uroot -p holberton > tmp_res ; head tmp_res
Enter password: 
origin  nb_fans
USA 99349
Sweden  47169
Finland 32878
United Kingdom  32518
Germany 29486
Norway  22405
Canada  8874
The Netherlands 8819
Italy   7178
ubuntu:~$ 
```

## Task16 - Old school band 
Write a SQL script that lists all bands with Glam rock as their main style, ranked by their longevity<br>
<br>
Requirements:<br>
* Import this table dump: metal_bands.sql.zip
* Column names must be: band_name and lifespan (in years)
* You should use attributes formed and split for computing the lifespan
* Your script can be executed on any database

```
ubuntu:~$ cat metal_bands.sql | mysql -uroot -p holberton
Enter password: 
ubuntu:~$ 
ubuntu:~$ cat 16-glam_rock.sql | mysql -uroot -p holberton 
Enter password: 
band_name   lifespan
Alice Cooper    56
Mötley Crüe   34
Marilyn Manson  31
The 69 Eyes 30
Hardcore Superstar  23
Nasty Idols 0
Hanoi Rocks 0
ubuntu:~$ 
```

## Task17 17 - Buy buy buy
Write a SQL script that creates a trigger that decreases the quantity of an item after adding a new order.<br>
Quantity in the table items can be negative.<br>

```
ubuntu:~$ cat 17-init.sql
-- Initial
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS orders;

CREATE TABLE IF NOT EXISTS items (
    name VARCHAR(255) NOT NULL,
    quantity int NOT NULL DEFAULT 10
);

CREATE TABLE IF NOT EXISTS orders (
    item_name VARCHAR(255) NOT NULL,
    number int NOT NULL
);

INSERT INTO items (name) VALUES ("apple"), ("pineapple"), ("pear");

ubuntu:~$ 
ubuntu:~$ cat 17-init.sql | mysql -uroot -p holberton 
Enter password: 
ubuntu:~$ 
ubuntu:~$ cat 17-store.sql | mysql -uroot -p holberton 
Enter password: 
ubuntu:~$ 
ubuntu:~$ cat 17-main.sql
Enter password: 
-- Show and add orders
SELECT * FROM items;
SELECT * FROM orders;

INSERT INTO orders (item_name, number) VALUES ('apple', 1);
INSERT INTO orders (item_name, number) VALUES ('apple', 3);
INSERT INTO orders (item_name, number) VALUES ('pear', 2);

SELECT "--";

SELECT * FROM items;
SELECT * FROM orders;

ubuntu:~$ 
ubuntu:~$ cat 17-main.sql | mysql -uroot -p holberton 
Enter password: 
name    quantity
apple   10
pineapple   10
pear    10
--
--
name    quantity
apple   6
pineapple   10
pear    8
item_name   number
apple   1
apple   3
pear    2
ubuntu:~$ 
```

## Task18 - Email validation to sent
Write a SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed.

```
ubuntu:~$ cat 18-init.sql
-- Initial
DROP TABLE IF EXISTS users;

CREATE TABLE IF NOT EXISTS users (
    id int not null AUTO_INCREMENT,
    email varchar(255) not null,
    name varchar(255),
    valid_email boolean not null default 0,
    PRIMARY KEY (id)
);

INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Bob");
INSERT INTO users (email, name, valid_email) VALUES ("sylvie@dylan.com", "Sylvie", 1);
INSERT INTO users (email, name, valid_email) VALUES ("jeanne@dylan.com", "Jeanne", 1);

ubuntu:~$ 
ubuntu:~$ cat 18-init.sql | mysql -uroot -p holberton 
Enter password: 
ubuntu:~$ 
ubuntu:~$ cat 18-valid_email.sql | mysql -uroot -p holberton 
Enter password: 
ubuntu:~$ 
ubuntu:~$ cat 18-main.sql
Enter password: 
-- Show users and update (or not) email
SELECT * FROM users;

UPDATE users SET valid_email = 1 WHERE email = "bob@dylan.com";
UPDATE users SET email = "sylvie+new@dylan.com" WHERE email = "sylvie@dylan.com";
UPDATE users SET name = "Jannis" WHERE email = "jeanne@dylan.com";

SELECT "--";
SELECT * FROM users;

UPDATE users SET email = "bob@dylan.com" WHERE email = "bob@dylan.com";

SELECT "--";
SELECT * FROM users;

ubuntu:~$ 
ubuntu:~$ cat 18-main.sql | mysql -uroot -p holberton 
Enter password: 
id  email   name    valid_email
1   bob@dylan.com   Bob 0
2   sylvie@dylan.com    Sylvie  1
3   jeanne@dylan.com    Jeanne  1
--
--
id  email   name    valid_email
1   bob@dylan.com   Bob 1
2   sylvie+new@dylan.com    Sylvie  0
3   jeanne@dylan.com    Jannis  1
--
--
id  email   name    valid_email
1   bob@dylan.com   Bob 1
2   sylvie+new@dylan.com    Sylvie  0
3   jeanne@dylan.com    Jannis  1
ubuntu:~$ 
```

## Task19 - Add bonus
Write a SQL script that creates a stored procedure AddBonus that adds a new correction for a student.<br>
<br>
Requirements:<br>
* Procedure AddBonus is taking 3 inputs (in this order):
* user_id, a users.id value (you can assume user_id is linked to an existing users)
* project_name, a new or already exists projects - if no projects.name found in the table, you should create it
* score, the score value for the correction

```
ubuntu:~$ cat 19-init.sql
-- Initial
DROP TABLE IF EXISTS corrections;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS projects;

CREATE TABLE IF NOT EXISTS users (
    id int not null AUTO_INCREMENT,
    name varchar(255) not null,
    average_score float default 0,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS projects (
    id int not null AUTO_INCREMENT,
    name varchar(255) not null,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS corrections (
    user_id int not null,
    project_id int not null,
    score int default 0,
    KEY `user_id` (`user_id`),
    KEY `project_id` (`project_id`),
    CONSTRAINT fk_user_id FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
    CONSTRAINT fk_project_id FOREIGN KEY (`project_id`) REFERENCES `projects` (`id`) ON DELETE CASCADE
);

INSERT INTO users (name) VALUES ("Bob");
SET @user_bob = LAST_INSERT_ID();

INSERT INTO users (name) VALUES ("Jeanne");
SET @user_jeanne = LAST_INSERT_ID();

INSERT INTO projects (name) VALUES ("C is fun");
SET @project_c = LAST_INSERT_ID();

INSERT INTO projects (name) VALUES ("Python is cool");
SET @project_py = LAST_INSERT_ID();


INSERT INTO corrections (user_id, project_id, score) VALUES (@user_bob, @project_c, 80);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_bob, @project_py, 96);

INSERT INTO corrections (user_id, project_id, score) VALUES (@user_jeanne, @project_c, 91);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_jeanne, @project_py, 73);

ubuntu:~$ 
ubuntu:~$ cat 19-init.sql | mysql -uroot -p holberton 
Enter password: 
ubuntu:~$ 
ubuntu:~$ cat 19-bonus.sql | mysql -uroot -p holberton 
Enter password: 
ubuntu:~$ 
ubuntu:~$ cat 19-main.sql
Enter password: 
-- Show and add bonus correction
SELECT * FROM projects;
SELECT * FROM corrections;

SELECT "--";

CALL AddBonus((SELECT id FROM users WHERE name = "Jeanne"), "Python is cool", 100);

CALL AddBonus((SELECT id FROM users WHERE name = "Jeanne"), "Bonus project", 100);
CALL AddBonus((SELECT id FROM users WHERE name = "Bob"), "Bonus project", 10);

CALL AddBonus((SELECT id FROM users WHERE name = "Jeanne"), "New bonus", 90);

SELECT "--";

SELECT * FROM projects;
SELECT * FROM corrections;

ubuntu:~$ 
ubuntu:~$ cat 19-main.sql | mysql -uroot -p holberton 
Enter password: 
id  name
1   C is fun
2   Python is cool
user_id project_id  score
1   1   80
1   2   96
2   1   91
2   2   73
--
--
--
--
id  name
1   C is fun
2   Python is cool
3   Bonus project
4   New bonus
user_id project_id  score
1   1   80
1   2   96
2   1   91
2   2   73
2   2   100
2   3   100
1   3   10
2   4   90
ubuntu:~$ 
```

## Task20 - Average score
Write a SQL script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.<br>
<br>
Requirements:
* Procedure ComputeAverageScoreForUser is taking 1 input:
* user_id, a users.id value (you can assume user_id is linked to an existing users)

```
ubuntu:~$ cat 20-init.sql
-- Initial
DROP TABLE IF EXISTS corrections;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS projects;

CREATE TABLE IF NOT EXISTS users (
    id int not null AUTO_INCREMENT,
    name varchar(255) not null,
    average_score float default 0,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS projects (
    id int not null AUTO_INCREMENT,
    name varchar(255) not null,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS corrections (
    user_id int not null,
    project_id int not null,
    score int default 0,
    KEY `user_id` (`user_id`),
    KEY `project_id` (`project_id`),
    CONSTRAINT fk_user_id FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
    CONSTRAINT fk_project_id FOREIGN KEY (`project_id`) REFERENCES `projects` (`id`) ON DELETE CASCADE
);

INSERT INTO users (name) VALUES ("Bob");
SET @user_bob = LAST_INSERT_ID();

INSERT INTO users (name) VALUES ("Jeanne");
SET @user_jeanne = LAST_INSERT_ID();

INSERT INTO projects (name) VALUES ("C is fun");
SET @project_c = LAST_INSERT_ID();

INSERT INTO projects (name) VALUES ("Python is cool");
SET @project_py = LAST_INSERT_ID();


INSERT INTO corrections (user_id, project_id, score) VALUES (@user_bob, @project_c, 80);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_bob, @project_py, 96);

INSERT INTO corrections (user_id, project_id, score) VALUES (@user_jeanne, @project_c, 91);
INSERT INTO corrections (user_id, project_id, score) VALUES (@user_jeanne, @project_py, 73);

ubuntu:~$ 
ubuntu:~$ cat 20-init.sql | mysql -uroot -p holberton 
Enter password: 
ubuntu:~$ 
ubuntu:~$ cat 20-average_score.sql | mysql -uroot -p holberton 
Enter password: 
ubuntu:~$ 
ubuntu:~$ cat 20-main.sql
-- Show and compute average score
SELECT * FROM users;
SELECT * FROM corrections;

SELECT "--";
CALL ComputeAverageScoreForUser((SELECT id FROM users WHERE name = "Jeanne"));

SELECT "--";
SELECT * FROM users;

ubuntu:~$ 
ubuntu:~$ cat 20-main.sql | mysql -uroot -p holberton 
Enter password: 
id  name    average_score
1   Bob 0
2   Jeanne  0
user_id project_id  score
1   1   80
1   2   96
2   1   91
2   2   73
--
--
--
--
id  name    average_score
1   Bob 0
2   Jeanne  82
ubuntu:~$ 
```
