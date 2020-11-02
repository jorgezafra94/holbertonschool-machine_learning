-- creating a trigger when change email
-- https://dba.stackexchange.com/questions/120078/update-one-mysql-column-when-another-is-edited
DELIMITER $$
CREATE TRIGGER reset_valid
    BEFORE UPDATE
    ON users FOR EACH ROW
BEGIN
    IF OLD.email <> NEW.email THEN
       SET NEW.valid_email = 0;
    END IF;
END $$

DELIMITER ;
