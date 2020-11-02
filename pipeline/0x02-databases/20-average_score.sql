-- creates another procedure in mysql
DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser (
    IN new_id INTEGER
    )
BEGIN
    UPDATE users SET average_score=(SELECT AVG(score) FROM corrections WHERE user_id = new_id)
    WHERE id=new_id;
END$$

DELIMITER ;
