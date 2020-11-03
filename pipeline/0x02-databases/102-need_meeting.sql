-- creating a view in mysql
-- https://www.mysqltutorial.org/create-sql-views-mysql.aspx#:~:text=%20MySQL%20CREATE%20VIEW%20%201%20Creating%20a,VIEW%20statement%20to%20create%20a%20view...%20More%20
CREATE VIEW need_meeting AS
    SELECT 
        name
    FROM
        students
    WHERE score < 80 AND
    (last_meeting IS NULL OR DATEDIFF(CURDATE(),last_meeting) > 30);
