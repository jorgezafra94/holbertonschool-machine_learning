-- creating a function in mysql
-- https://www.techonthenet.com/mysql/functions.php
-- https://www.mysqltutorial.org/mysql-stored-function/
DELIMITER $$

CREATE FUNCTION SafeDiv(a INT, b INT)

RETURNS FLOAT

BEGIN
	DECLARE result FLOAT;
	SET result = 0.0;
	IF b = 0 THEN
	   SET result = 0;
	ELSE
	   SET result = a / b;
	END IF;
	
	RETURN result;
END $$

DELIMITER ;
