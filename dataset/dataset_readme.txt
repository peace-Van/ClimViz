MD5 checksum for climate_data_land.h5: 4e4bd9db7d6042041f8b2377d0302ff3

The hdf5 file contains 7 datasets with 'lzf' compression

'indices' - shape (66501, 2) - latitude, longitude for all land pixels
'elev' - shape (66501, ) - average elevation (in metre) for each land pixel
'tmp' - shape (66501, 123, 12) - average daily mean temperature of each month during 1901-2023 for each land pixel
'pre' - shape (66501, 123, 12) - precipitation of each month during 1901-2023 for each land pixel
'pre' - shape (66501, 123, 12) - potential evapotranspiration (PET) of each month during 1901-2023 for each land pixel
'tmn' - shape (66501, 123, 12) - average daily minimum temperature of each month during 1901-2023 for each land pixel
'tmx' - shape (66501, 123, 12) - average daily maximum temperature of each month during 1901-2023 for each land pixel

Temperatures are in Celcius, precipitation and PET are in millimeters
'elev', 'tmn', 'tmx' are not used in the climate classification models, only for plotting

-------------------------------------

MD5 checksum for climate_variables.h5: 08807fe7c0730d81fbfceee48ffca85e

The hdf5 file contains 1 dataset with 'lzf' compression

'res' - shape (66502, 123, 10)
res[-1, :, :] is the global average
Ten climate variables during 1901-2023 for each land pixel

"Lowest Monthly Temperature"
"Highest Monthly Temperature"
"Highest Monthly Precipitation"
"Lowest Monthly Precipitation"
"Annual Mean Temperature"
"Annual Total Precipitation"
"Aridity Index"
"Cryohumidity"
"Continentality"
"Seasonality"

Temperature in Celcius, precipitation in millimeters
