MD5 checksum for climate_data_land.h5: d103c466c9793cf05d223a0b7c089887

The hdf5 file contains 7 datasets with gzip compression

'indices' - shape (66501, 2) - latitude, longitude for all land pixels
'elev' - shape (66501, ) - average elevation (in metre) for each land pixel
'tmp' - shape (66501, 124, 12) - average daily mean temperature of each month during 1901-2024 for each land pixel
'pre' - shape (66501, 124, 12) - precipitation of each month during 1901-2024 for each land pixel
'pet' - shape (66501, 124, 12) - potential evapotranspiration (PET) of each month during 1901-2024 for each land pixel
'tmn' - shape (66501, 124, 12) - average daily minimum temperature of each month during 1901-2024 for each land pixel
'tmx' - shape (66501, 124, 12) - average daily maximum temperature of each month during 1901-2024 for each land pixel

Temperatures are in Celcius, precipitation and PET are in millimeters
'elev', 'tmp', 'pet' are not used in the climate classification models, only for plotting

-------------------------------------

MD5 checksum for climate_variables.h5: 0dfedc2853882e0295cc68f5c201af7f

The hdf5 file contains 1 dataset with gzip compression

'res' - shape (66502, 124, 10)
res[-1, :, :] is the global average
Ten climate variables during 1901-2024 for each land pixel

"Coldest Month Mean Temperature": 0,
"Hottest Month Mean Temperature": 1,
"Coldest Month Mean Daily Minimum": 2,
"Hottest Month Mean Daily Maximum": 3,
"Mean Annual Temperature": 4,
"Wettest Month Precipitation": 5,
"Driest Month Precipitation": 6,
"Total Annual Precipitation": 7,
"Thermal Index": 8,
"Aridity Index": 9,

Temperature in Celcius, precipitation in millimeters
