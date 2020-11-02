-- getting max value of each state in temperature table
SELECT state, MAX(value) from temperatures
GROUP BY state
ORDER BY state;
