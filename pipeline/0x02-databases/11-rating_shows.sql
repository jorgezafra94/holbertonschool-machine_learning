-- remember always create the database
-- then you can mysql -u root -p hbtn_0d_tvshows_rate < ./hbtn_0d_tvshows_rate.sql
SELECT tv_shows.title AS title, SUM(tv_show_ratings.rate) AS rating FROM tv_shows
LEFT JOIN tv_show_ratings ON tv_show_ratings.show_id = tv_shows.id
GROUP BY title
ORDER BY rating DESC; 
