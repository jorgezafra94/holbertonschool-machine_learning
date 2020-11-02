-- getting max value of each state in temperature table
SELECT state, MAX(value) AS max_temp from temperatures
GROUP BY state
ORDER BY state;
