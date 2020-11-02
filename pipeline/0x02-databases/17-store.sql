-- learning for first time Triggers
-- got the infor from https://www.mysqltutorial.org/mysql-triggers/mysql-after-insert-trigger/
-- the  idea is to use the name to know what quantity we should decrease
-- so here if a new order is created the number of the elements in the order will be decreased in the items table
-- like an inventory if you sell items you have to decrease the number of elements that you have
DELIMITER //
CREATE TRIGGER items_updated
   AFTER INSERT
   ON orders FOR EACH ROW
BEGIN
   UPDATE items SET quantity = quantity - NEW.number
   WHERE items.name = NEW.item_name;
END //

DELIMITER ;
