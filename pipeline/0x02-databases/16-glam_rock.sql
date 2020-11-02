-- playing with metal_band table getting glam style bands
SELECT band_name,
       IF(split is NULL, YEAR(CURDATE()), split) - formed AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC, band_name DESC;
