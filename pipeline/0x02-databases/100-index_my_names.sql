-- script that creates an index idx_name_first
-- on the table names and the first letter of name
-- https://www.mysqltutorial.org/mysql-index/mysql-create-index/#:~:text=%20MySQL%20CREATE%20INDEX%20%201%20The%20phone,the%20time%20of%20creation.%20For%20example%2C...%20More%20
CREATE INDEX idx_name_first ON names (name(1));
