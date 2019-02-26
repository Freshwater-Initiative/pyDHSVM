################################################################################
# DHSVM INPUT FILE FORMAT
################################################################################
# The file is organized in sections [...], which contain key = entry pairs.  
# The file is free format, in that correct reading of the file is not dependent 
# on spaces and/or the order of the key-entry pairs within a section.  
# The keys are not case-sensitive, but the entries are, because filenames on a 
# UNIX platform are case-sensitive.  
# Comments are preceded by a '#', and run from the occurrence of '#' till the 
# end of the line. You can comment out an entire line (like in this 
# header), or you can place a comment after an entry.  
# It is important to place the key-entry pair in the correct section, since it 
# will not be found if it is in another section.  
# The easiest way to make the input file is to fill out this default template.  
# Since DHSVM will only use the keys that it requires you do not have to worry 
# about empty entries for keys that are not needed. For example, if you are 
# running the model in point mode, you do not have to fill out the routing 
# section.  If you have already filled it out you can leave it, since DHSVM will 
# not use the information.  This allows easy switching between point and basin 
# mode.  
# For more information about the specific entries see the DHSVM web page


################################################################################
# OPTIONS SECTION 
################################################################################
[OPTIONS]                                 # Model Options
Format               = BIN                # BIN, BYTESWAP or NETCDF
Extent               = BASIN              # POINT or BASIN
Gradient             = TOPOGRAPHY 	  # TOPOGRAPHY or WATERTABLE
Flow Routing         = NETWORK            # UNIT_HYDROGRAPH or NETWORK
Sensible Heat Flux   = FALSE              # TRUE or FALSE
Sediment	     = FALSE		  # TRUE or FALSE
Sediment Input File  =			  # path for sediment configuration file
Overland Routing     = CONVENTIONAL       # CONVENTIONAL or KINEMATIC
Infiltration	     = DYNAMIC		  # Static or Dynamic
Interpolation        = VARCRESS           # NEAREST or INVDIST or VARCRESS
MM5                  = FALSE		  # TRUE or FALSE
QPF                  = FALSE              # TRUE or FALSE
PRISM                = FALSE               # TRUE or FALSE
PRISM data path      = PRISM/skagit800m # path for PRISM files
PRISM data extension = bin                # file extension for PRISM files
Canopy radiation attenuation mode = FIXED # FIXED or VARIABLE
Shading              = TRUE   	          # TRUE or FALSE
Shading data path    = solar/Shadow
Shading data extension = bin             # file extension for shading files 
Skyview data path    = solar/SkyView.bin
Snotel               = FALSE               # TRUE or FALSE
Outside              = TRUE	          # TRUE or FALSE
Rhoverride           = TRUE              # TRUE or FALSE
Precipitation Source = STATION            # STATION or RADAR
Wind Source          = STATION            # STATION or MODEL
Temperature lapse rate = CONSTANT         # CONSTANT or VARIABLE
Precipitation lapse rate = CONSTANT       # CONSTANT, MAP, or VARIABLE
Cressman radius      = 10                 # in model pixels
Cressman stations    =  6                 # number of stations
Water Quality        = FALSE              # TRUE or FALSE
Antecedent Dry Days  = 0.1                # For water quality option
GLACIER              =  GLDYNAMIC         # NO_GLACIER, GLSPINUP, GLSTATIC,GLDYNAMIC
Stream Temperature   = FALSE              # TRUE or FALSE
Canopy Shading       = TRUE             # If false, only topo shading is considered
RIPARIAN Shading       = FALSE              # If false, only topo shading is considered
GRIDDED MET DATA = FALSE
IMPROVED RADIATION SCHEME = FALSE
################################################################################
# MODEL AREA SECTION
################################################################################
[AREA]                                    # Model area
Coordinate System    = UTM                # UTM or USER_DEFINED
Extreme North        = 5464395.000            # Coordinate for northern edge of grid
Extreme West         = 535665.000             # Coordinate for western edge of grid
Center Latitude      = 48.633              # Central parallel of basin 
Center Longitude     = -121.57            # Central meridian of basin 
Time Zone Meridian   = -120.0             # Time zone meridian for area 
Number of Rows       = 1020                # Number of rows
Number of Columns    = 916               # Number of columns
Grid spacing         = 150                # Grid resolution in m 
Point North          =                    # North coordinate for point model if Extent = POINT 
Point East           =                    # East coordinate for point modelif Extent = POINT 

################################################################################
# TIME SECTION
################################################################################
[TIME]                                    # Model period
Time Step            = 3                  # Model time step (hours)
Model Start          = 10/01/1960-00     #Model start time (MM/DD/YYYY-HH)
Model End            = 12/31/2011-00      # Model end time (MM/DD/YYYY-HH)
#Model Start          = 08/01/1998-00      # Model start time (MM/DD/YYYY-HH)
#Model End            = 09/01/1998-00      # Model end time (MM/DD/YYYY-HH)
################################################################################
# CONSTANTS SECTION
################################################################################
[CONSTANTS]                               # Model constants
Ground Roughness     = 0.02               # Roughness of soil surface (m)
Snow Roughness = 0.001498 # Roughness of snow surface (m)
Rain Threshold       = -0.5               # Minimum temperature at which rain occurs (C)
Snow Threshold       = 0.5               # Maximum temperature at which snow occurs (C)
Snow Water Capacity  = 0.03               # Snow liquid water holding capacity(fraction)
Reference Height     = 50.0               # Reference height (m)
Rain LAI Multiplier  = 0.0001             # LAI Multiplier for rain interception
Snow LAI Multiplier  = 0.005             # LAI Mulitplier for snow interception
Min Intercepted Snow = 0.005              # Intercepted snow that can only be melted (m)
Outside Basin Value  = 0                  # Value in mask that indicates outside the basin
Temperature Lapse Rate = -0.0048 # Temperature lapse rate (C/m)
Precipitation Lapse Rate =  0.0006    	# Precipitation lapse rate (m/m)
Glacier Albedo = 0.333665 #Glacier Albedo
Maximum Snow Albedo = 0.878114 #Maximum Snow Albedo      
Precipitation Multiplier = 0.00001
Snowslide Parameter1 = 700.
Snowslide Parameter2 = 0.12
ALBEDO ACCUMULATION LAMBDA = 0.92
ALBEDO MELTING LAMBDA = 0.70
ALBEDO ACCUMULATION MIN = 0.4
ALBEDO MELTING MIN = 0.4
################################################################################
# TERRAIN INFORMATION SECTION
################################################################################
[TERRAIN]                                 # Terrain information
DEM File             = arc/demsrtm150f.bin# path for DEM file
Basin Mask File      = arc/skagit.mask.bin # path for mask file
Domain Mask File   = arc/skagit.mask.bin #domain_northeast.bin  #domain.bin
################################################################################
# GLACIER MODEL SECTION #The following block only needs to be filled out if GLACIER = TRUE.
################################################################################
[GLACIER]
Domain Mask File = arc/skagit.mask.bin #skagit_mask2015.bin #domain_northeast.bin
Glacier Mask File      =  arc/noisy6_silver7.bin ##glacier_mask1987.bin  # path for glacier extent file, required only if GLACIER = TRUE
Glacier Bed Topo File =     arc/demsrtm150f.bin #bedtopo10.bin   #path for glacier bedtopography file, required only if GLACIER = TRUE or SPINUP
Glacier Surface Topo File =   arc/surftopo1960.bin # dem1000_1987.bin     #arc/dem1000.bin        #    path for glacier surface topography file, required only if GLACIER = TRUE
Basin Mask File      =         arc/skagit.mask.bin         # path for basin mask file,  required only if GLACIER = TRUE
Glacier Mass Balance File =   arc/dem1000_1987.bin  #balance_sum.bin        # path for glacier annual mass balance file, required only if GLACIER = SPINUP

################################################################################

# ROUTING SECTION
################################################################################
[ROUTING]                                 # Routing information. This section is 
                                          # only relevant if the Extent = BASIN

################ STREAM NETWORK ################################################
# The following three fields are only used if Flow Routing = NETWORK

Stream Map File      =  network/new.mapfile # path for stream map file
Stream Network File  = network/NetworkFile_update2017.txt # path for stream network file
Stream Class File    = network/adjust.classfile # path for stream class file
Riparian Veg File    = network/rveg.baseline.network
################ ROAD NETWORK ##################################################
# The following three fields are only used if Flow Routing = NETWORK and there
# is a road network

#Road Map File        =                    # path for road map file
#Road Network File    =                    # path for road network file
#Road Class File      =                    # path for road network file

################ UNIT HYDROGRAPH ###############################################
# The following two fields are only used if Flow Routing = UNIT_HYDROGRAPH

Travel Time File     =                    # path for travel time file
Unit Hydrograph File =                    # path for unit hydrograph file

###############################################################################
# METEOROLOGY SECTION 
################################################################################
[METEOROLOGY]                             # Meteorological stations

#Grid Rows = 14
#Grid Cols = 27  
#Extreme North Lat = 48.484375
#Extreme East Lon = -120.71875
#Grid Size = 0.0625
#Met file path = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_

Number of Stations = 241                       # Number of meteorological stations

Station Name 1 = data_49.28125_-120.84375
North Coordinate 1 = 5460959.728000
East Coordinate 1 = 656818.179400
Elevation 1 = 1727
Station File 1 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.28125_-120.84375

Station Name 2 = data_49.28125_-120.90625
North Coordinate 2 = 5460831.854000
East Coordinate 2 = 652273.028100
Elevation 2 = 1551
Station File 2 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.28125_-120.90625

Station Name 3 = data_49.28125_-121.03125
North Coordinate 3 = 5460587.396000
East Coordinate 3 = 643182.646200
Elevation 3 = 1526
Station File 3 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.28125_-121.03125

Station Name 4 = data_49.21875_-120.90625
North Coordinate 4 = 5453884.405000
East Coordinate 4 = 652465.444300
Elevation 4 = 1604
Station File 4 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.21875_-120.90625

Station Name 5 = data_49.21875_-120.96875
North Coordinate 5 = 5453760.254000
East Coordinate 5 = 647914.518700
Elevation 5 = 1317
Station File 5 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.21875_-120.96875

Station Name 6 = data_49.21875_-121.03125
North Coordinate 6 = 5453639.868000
East Coordinate 6 = 643363.567500
Elevation 6 = 1275
Station File 6 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.21875_-121.03125

Station Name 7 = data_49.15625_-120.90625
North Coordinate 7 = 5446937.022000
East Coordinate 7 = 652657.677700
Elevation 7 = 1513
Station File 7 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.15625_-120.90625

Station Name 8 = data_49.15625_-120.96875
North Coordinate 8 = 5446812.832000
East Coordinate 8 = 648101.009900
Elevation 8 = 1441
Station File 8 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.15625_-120.96875

Station Name 9 = data_49.15625_-121.03125
North Coordinate 9 = 5446692.407000
East Coordinate 9 = 643544.316900
Elevation 9 = 1575
Station File 9 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.15625_-121.03125

Station Name 10 = data_49.15625_-121.09375
North Coordinate 10 = 5446575.748000
East Coordinate 10 = 638987.599500
Elevation 10 = 1289
Station File 10 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.15625_-121.09375

Station Name 11 = data_49.15625_-121.15625
North Coordinate 11 = 5446462.855000
East Coordinate 11 = 634430.858300
Elevation 11 = 1068
Station File 11 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.15625_-121.15625

Station Name 12 = data_49.15625_-121.21875
North Coordinate 12 = 5446353.726000
East Coordinate 12 = 629874.094200
Elevation 12 = 1448
Station File 12 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.15625_-121.21875

Station Name 13 = data_49.15625_-121.28125
North Coordinate 13 = 5446248.363000
East Coordinate 13 = 625317.307900
Elevation 13 = 1334
Station File 13 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.15625_-121.28125

Station Name 14 = data_49.09375_-120.96875
North Coordinate 14 = 5439865.475000
East Coordinate 14 = 648287.323600
Elevation 14 = 1530
Station File 14 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.09375_-120.96875

Station Name 15 = data_49.09375_-121.03125
North Coordinate 15 = 5439745.013000
East Coordinate 15 = 643724.894300
Elevation 15 = 1372
Station File 15 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.09375_-121.03125

Station Name 16 = data_49.09375_-121.09375
North Coordinate 16 = 5439628.317000
East Coordinate 16 = 639162.440800
Elevation 16 = 974
Station File 16 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.09375_-121.09375

Station Name 17 = data_49.09375_-121.15625
North Coordinate 17 = 5439515.388000
East Coordinate 17 = 634599.963900
Elevation 17 = 936
Station File 17 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.09375_-121.15625

Station Name 18 = data_49.09375_-121.21875
North Coordinate 18 = 5439406.226000
East Coordinate 18 = 630037.464400
Elevation 18 = 1363
Station File 18 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.09375_-121.21875

Station Name 19 = data_49.09375_-121.28125
North Coordinate 19 = 5439300.829000
East Coordinate 19 = 625474.943100
Elevation 19 = 1379
Station File 19 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.09375_-121.28125

Station Name 20 = data_49.03125_-120.84375
North Coordinate 20 = 5433170.488000
East Coordinate 20 = 657609.705500
Elevation 20 = 1719
Station File 20 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.03125_-120.84375

Station Name 21 = data_49.03125_-120.90625
North Coordinate 21 = 5433042.452000
East Coordinate 21 = 653041.595200
Elevation 21 = 1589
Station File 21 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.03125_-120.90625

Station Name 22 = data_49.03125_-120.96875
North Coordinate 22 = 5432918.184000
East Coordinate 22 = 648473.459500
Elevation 22 = 1534
Station File 22 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.03125_-120.96875

Station Name 23 = data_49.03125_-121.03125
North Coordinate 23 = 5432797.685000
East Coordinate 23 = 643905.299300
Elevation 23 = 1071
Station File 23 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.03125_-121.03125

Station Name 24 = data_49.03125_-121.09375
North Coordinate 24 = 5432680.953000
East Coordinate 24 = 639337.115200
Elevation 24 = 775
Station File 24 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.03125_-121.09375

Station Name 25 = data_49.03125_-121.15625
North Coordinate 25 = 5432567.989000
East Coordinate 25 = 634768.908100
Elevation 25 = 1393
Station File 25 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.03125_-121.15625

Station Name 26 = data_49.03125_-121.21875
North Coordinate 26 = 5432458.793000
East Coordinate 26 = 630200.678800
Elevation 26 = 1633
Station File 26 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.03125_-121.21875

Station Name 27 = data_49.03125_-121.28125
North Coordinate 27 = 5432353.364000
East Coordinate 27 = 625632.427900
Elevation 27 = 1590
Station File 27 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_49.03125_-121.28125

Station Name 28 = data_48.96875_-120.90625
North Coordinate 28 = 5426095.265000
East Coordinate 28 = 653233.278900
Elevation 28 = 1719
Station File 28 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.96875_-120.90625

Station Name 29 = data_48.96875_-120.96875
North Coordinate 29 = 5425970.960000
East Coordinate 29 = 648659.417400
Elevation 29 = 1343
Station File 29 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.96875_-120.96875

Station Name 30 = data_48.96875_-121.03125
North Coordinate 30 = 5425850.424000
East Coordinate 30 = 644085.531700
Elevation 30 = 1190
Station File 30 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.96875_-121.03125

Station Name 31 = data_48.96875_-121.09375
North Coordinate 31 = 5425733.656000
East Coordinate 31 = 639511.622600
Elevation 31 = 767
Station File 31 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.96875_-121.09375

Station Name 32 = data_48.96875_-121.15625
North Coordinate 32 = 5425620.658000
East Coordinate 32 = 634937.690800
Elevation 32 = 1445
Station File 32 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.96875_-121.15625

Station Name 33 = data_48.96875_-121.21875
North Coordinate 33 = 5425511.429000
East Coordinate 33 = 630363.737000
Elevation 33 = 1764
Station File 33 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.96875_-121.21875

Station Name 34 = data_48.90625_-120.78125
North Coordinate 34 = 5419408.142000
East Coordinate 34 = 662583.867900
Elevation 34 = 1823
Station File 34 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.90625_-120.78125

Station Name 35 = data_48.90625_-120.84375
North Coordinate 35 = 5419276.257000
East Coordinate 35 = 658004.336000
Elevation 35 = 1450
Station File 35 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.90625_-120.84375

Station Name 36 = data_48.90625_-120.90625
North Coordinate 36 = 5419148.144000
East Coordinate 36 = 653424.778800
Elevation 36 = 1444
Station File 36 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.90625_-120.90625

Station Name 37 = data_48.90625_-120.96875
North Coordinate 37 = 5419023.801000
East Coordinate 37 = 648845.197100
Elevation 37 = 1109
Station File 37 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.90625_-120.96875

Station Name 38 = data_48.90625_-121.03125
North Coordinate 38 = 5418903.229000
East Coordinate 38 = 644265.591500
Elevation 38 = 977
Station File 38 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.90625_-121.03125

Station Name 39 = data_48.90625_-121.09375
North Coordinate 39 = 5418786.427000
East Coordinate 39 = 639685.962700
Elevation 39 = 971
Station File 39 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.90625_-121.09375

Station Name 40 = data_48.90625_-121.15625
North Coordinate 40 = 5418673.395000
East Coordinate 40 = 635106.311600
Elevation 40 = 1196
Station File 40 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.90625_-121.15625

Station Name 41 = data_48.90625_-121.21875
North Coordinate 41 = 5418564.133000
East Coordinate 41 = 630526.638900
Elevation 41 = 1309
Station File 41 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.90625_-121.21875

Station Name 42 = data_48.90625_-121.28125
North Coordinate 42 = 5418458.640000
East Coordinate 42 = 625946.945200
Elevation 42 = 1392
Station File 42 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.90625_-121.28125

Station Name 43 = data_48.84375_-120.78125
North Coordinate 43 = 5412461.163000
East Coordinate 43 = 662786.614700
Elevation 43 = 1765
Station File 43 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-120.78125

Station Name 44 = data_48.84375_-120.84375
North Coordinate 44 = 5412329.240000
East Coordinate 44 = 658201.367300
Elevation 44 = 1635
Station File 44 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-120.84375

Station Name 45 = data_48.84375_-120.90625
North Coordinate 45 = 5412201.089000
East Coordinate 45 = 653616.094900
Elevation 45 = 1553
Station File 45 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-120.90625

Station Name 46 = data_48.84375_-120.96875
North Coordinate 46 = 5412076.709000
East Coordinate 46 = 649030.798300
Elevation 46 = 1442
Station File 46 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-120.96875

Station Name 47 = data_48.84375_-121.03125
North Coordinate 47 = 5411956.102000
East Coordinate 47 = 644445.478300
Elevation 47 = 689
Station File 47 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-121.03125

Station Name 48 = data_48.84375_-121.09375
North Coordinate 48 = 5411839.265000
East Coordinate 48 = 639860.135400
Elevation 48 = 1391
Station File 48 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-121.09375

Station Name 49 = data_48.84375_-121.15625
North Coordinate 49 = 5411726.200000
East Coordinate 49 = 635274.770500
Elevation 49 = 1483
Station File 49 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-121.15625

Station Name 50 = data_48.84375_-121.21875
North Coordinate 50 = 5411616.905000
East Coordinate 50 = 630689.384200
Elevation 50 = 1210
Station File 50 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-121.21875

Station Name 51 = data_48.84375_-121.28125
North Coordinate 51 = 5411511.382000
East Coordinate 51 = 626103.977400
Elevation 51 = 1548
Station File 51 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-121.28125

Station Name 52 = data_48.84375_-121.34375
North Coordinate 52 = 5411409.629000
East Coordinate 52 = 621518.550600
Elevation 52 = 1737
Station File 52 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-121.34375

Station Name 53 = data_48.84375_-121.40625
North Coordinate 53 = 5411311.646000
East Coordinate 53 = 616933.104700
Elevation 53 = 1292
Station File 53 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-121.40625

Station Name 54 = data_48.84375_-121.46875
North Coordinate 54 = 5411217.433000
East Coordinate 54 = 612347.640400
Elevation 54 = 1253
Station File 54 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.84375_-121.46875

Station Name 55 = data_48.78125_-120.78125
North Coordinate 55 = 5405514.249000
East Coordinate 55 = 662989.166500
Elevation 55 = 1454
Station File 55 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-120.78125

Station Name 56 = data_48.78125_-120.84375
North Coordinate 56 = 5405382.287000
East Coordinate 56 = 658398.209000
Elevation 56 = 1739
Station File 56 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-120.84375

Station Name 57 = data_48.78125_-120.90625
North Coordinate 57 = 5405254.099000
East Coordinate 57 = 653807.226900
Elevation 57 = 1436
Station File 57 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-120.90625

Station Name 58 = data_48.78125_-120.96875
North Coordinate 58 = 5405129.684000
East Coordinate 58 = 649216.221000
Elevation 58 = 1695
Station File 58 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-120.96875

Station Name 59 = data_48.78125_-121.03125
North Coordinate 59 = 5405009.041000
East Coordinate 59 = 644625.191900
Elevation 59 = 724
Station File 59 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.03125

Station Name 60 = data_48.78125_-121.09375
North Coordinate 60 = 5404892.171000
East Coordinate 60 = 640034.140400
Elevation 60 = 972
Station File 60 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.09375

Station Name 61 = data_48.78125_-121.15625
North Coordinate 61 = 5404779.072000
East Coordinate 61 = 635443.067100
Elevation 61 = 1246
Station File 61 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.15625

Station Name 62 = data_48.78125_-121.21875
North Coordinate 62 = 5404669.746000
East Coordinate 62 = 630851.972900
Elevation 62 = 1265
Station File 62 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.21875

Station Name 63 = data_48.78125_-121.28125
North Coordinate 63 = 5404564.192000
East Coordinate 63 = 626260.858400
Elevation 63 = 1574
Station File 63 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.28125

Station Name 64 = data_48.78125_-121.34375
North Coordinate 64 = 5404462.410000
East Coordinate 64 = 621669.724300
Elevation 64 = 1408
Station File 64 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.34375

Station Name 65 = data_48.78125_-121.40625
North Coordinate 65 = 5404364.398000
East Coordinate 65 = 617078.571300
Elevation 65 = 1391
Station File 65 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.40625

Station Name 66 = data_48.78125_-121.46875
North Coordinate 66 = 5404270.158000
East Coordinate 66 = 612487.400100
Elevation 66 = 1194
Station File 66 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.46875

Station Name 67 = data_48.78125_-121.53125
North Coordinate 67 = 5404179.689000
East Coordinate 67 = 607896.211600
Elevation 67 = 784
Station File 67 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.53125

Station Name 68 = data_48.78125_-121.59375
North Coordinate 68 = 5404092.991000
East Coordinate 68 = 603305.006200
Elevation 68 = 1108
Station File 68 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.59375

Station Name 69 = data_48.78125_-121.65625
North Coordinate 69 = 5404010.063000
East Coordinate 69 = 598713.784900
Elevation 69 = 820
Station File 69 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.65625

Station Name 70 = data_48.78125_-121.71875
North Coordinate 70 = 5403930.906000
East Coordinate 70 = 594122.548300
Elevation 70 = 1096
Station File 70 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.71875

Station Name 71 = data_48.78125_-121.78125
North Coordinate 71 = 5403855.520000
East Coordinate 71 = 589531.297000
Elevation 71 = 1802
Station File 71 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.78125_-121.78125

Station Name 72 = data_48.71875_-120.71875
North Coordinate 72 = 5398703.172000
East Coordinate 72 = 667788.160100
Elevation 72 = 1724
Station File 72 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-120.71875

Station Name 73 = data_48.71875_-120.78125
North Coordinate 73 = 5398567.399000
East Coordinate 73 = 663191.522900
Elevation 73 = 1591
Station File 73 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-120.78125

Station Name 74 = data_48.71875_-120.84375
North Coordinate 74 = 5398435.400000
East Coordinate 74 = 658594.860800
Elevation 74 = 1416
Station File 74 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-120.84375

Station Name 75 = data_48.71875_-120.90625
North Coordinate 75 = 5398307.175000
East Coordinate 75 = 653998.174500
Elevation 75 = 1369
Station File 75 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-120.90625

Station Name 76 = data_48.71875_-120.96875
North Coordinate 76 = 5398182.724000
East Coordinate 76 = 649401.464700
Elevation 76 = 1168
Station File 76 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-120.96875

Station Name 77 = data_48.71875_-121.03125
North Coordinate 77 = 5398062.047000
East Coordinate 77 = 644804.732100
Elevation 77 = 1139
Station File 77 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.03125

Station Name 78 = data_48.71875_-121.09375
North Coordinate 78 = 5397945.143000
East Coordinate 78 = 640207.977500
Elevation 78 = 867
Station File 78 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.09375

Station Name 79 = data_48.71875_-121.15625
North Coordinate 79 = 5397832.013000
East Coordinate 79 = 635611.201400
Elevation 79 = 866
Station File 79 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.15625

Station Name 80 = data_48.71875_-121.21875
North Coordinate 80 = 5397722.656000
East Coordinate 80 = 631014.404700
Elevation 80 = 1111
Station File 80 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.21875

Station Name 81 = data_48.71875_-121.28125
North Coordinate 81 = 5397617.072000
East Coordinate 81 = 626417.588000
Elevation 81 = 1017
Station File 81 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.28125

Station Name 82 = data_48.71875_-121.34375
North Coordinate 82 = 5397515.260000
East Coordinate 82 = 621820.752000
Elevation 82 = 1236
Station File 82 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.34375

Station Name 83 = data_48.71875_-121.40625
North Coordinate 83 = 5397417.221000
East Coordinate 83 = 617223.897400
Elevation 83 = 1198
Station File 83 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.40625

Station Name 84 = data_48.71875_-121.46875
North Coordinate 84 = 5397322.954000
East Coordinate 84 = 612627.025000
Elevation 84 = 1470
Station File 84 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.46875

Station Name 85 = data_48.71875_-121.53125
North Coordinate 85 = 5397232.459000
East Coordinate 85 = 608030.135400
Elevation 85 = 1294
Station File 85 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.53125

Station Name 86 = data_48.71875_-121.59375
North Coordinate 86 = 5397145.736000
East Coordinate 86 = 603433.229300
Elevation 86 = 676
Station File 86 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.59375

Station Name 87 = data_48.71875_-121.65625
North Coordinate 87 = 5397062.785000
East Coordinate 87 = 598836.307400
Elevation 87 = 419
Station File 87 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.65625

Station Name 88 = data_48.71875_-121.71875
North Coordinate 88 = 5396983.606000
East Coordinate 88 = 594239.370400
Elevation 88 = 639
Station File 88 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.71875

Station Name 89 = data_48.71875_-121.78125
North Coordinate 89 = 5396908.198000
East Coordinate 89 = 589642.419100
Elevation 89 = 1118
Station File 89 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.78125

Station Name 90 = data_48.71875_-121.84375
North Coordinate 90 = 5396836.561000
East Coordinate 90 = 585045.454100
Elevation 90 = 1366
Station File 90 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.71875_-121.84375

Station Name 91 = data_48.65625_-120.78125
North Coordinate 91 = 5391620.613000
East Coordinate 91 = 663393.683700
Elevation 91 = 1845
Station File 91 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-120.78125

Station Name 92 = data_48.65625_-120.84375
North Coordinate 92 = 5391488.577000
East Coordinate 92 = 658791.322500
Elevation 92 = 1421
Station File 92 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-120.84375

Station Name 93 = data_48.65625_-120.90625
North Coordinate 93 = 5391360.316000
East Coordinate 93 = 654188.937500
Elevation 93 = 1626
Station File 93 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-120.90625

Station Name 94 = data_48.65625_-120.96875
North Coordinate 94 = 5391235.831000
East Coordinate 94 = 649586.529300
Elevation 94 = 1569
Station File 94 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-120.96875

Station Name 95 = data_48.65625_-121.03125
North Coordinate 95 = 5391115.120000
East Coordinate 95 = 644984.098800
Elevation 95 = 1176
Station File 95 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.03125

Station Name 96 = data_48.65625_-121.09375
North Coordinate 96 = 5390998.183000
East Coordinate 96 = 640381.646500
Elevation 96 = 1150
Station File 96 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.09375

Station Name 97 = data_48.65625_-121.15625
North Coordinate 97 = 5390885.022000
East Coordinate 97 = 635779.173100
Elevation 97 = 1807
Station File 97 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.15625

Station Name 98 = data_48.65625_-121.21875
North Coordinate 98 = 5390775.634000
East Coordinate 98 = 631176.679400
Elevation 98 = 937
Station File 98 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.21875

Station Name 99 = data_48.65625_-121.28125
North Coordinate 99 = 5390670.020000
East Coordinate 99 = 626574.166000
Elevation 99 = 652
Station File 99 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.28125

Station Name 100 = data_48.65625_-121.34375
North Coordinate 100 = 5390568.180000
East Coordinate 100 = 621971.633600
Elevation 100 = 945
Station File 100 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.34375

Station Name 101 = data_48.65625_-121.40625
North Coordinate 101 = 5390470.113000
East Coordinate 101 = 617369.083000
Elevation 101 = 931
Station File 101 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.40625

Station Name 102 = data_48.65625_-121.46875
North Coordinate 102 = 5390375.820000
East Coordinate 102 = 612766.514700
Elevation 102 = 1101
Station File 102 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.46875

Station Name 103 = data_48.65625_-121.53125
North Coordinate 103 = 5390285.300000
East Coordinate 103 = 608163.929600
Elevation 103 = 1403
Station File 103 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.53125

Station Name 104 = data_48.65625_-121.59375
North Coordinate 104 = 5390198.553000
East Coordinate 104 = 603561.328200
Elevation 104 = 1300
Station File 104 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.59375

Station Name 105 = data_48.65625_-121.65625
North Coordinate 105 = 5390115.578000
East Coordinate 105 = 598958.711300
Elevation 105 = 771
Station File 105 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.65625

Station Name 106 = data_48.65625_-121.71875
North Coordinate 106 = 5390036.377000
East Coordinate 106 = 594356.079600
Elevation 106 = 342
Station File 106 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.71875

Station Name 107 = data_48.65625_-121.78125
North Coordinate 107 = 5389960.948000
East Coordinate 107 = 589753.433700
Elevation 107 = 877
Station File 107 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.65625_-121.78125

Station Name 108 = data_48.59375_-120.71875
North Coordinate 108 = 5384809.741000
East Coordinate 108 = 668203.679500
Elevation 108 = 1898
Station File 108 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-120.71875

Station Name 109 = data_48.59375_-120.78125
North Coordinate 109 = 5384673.892000
East Coordinate 109 = 663595.648800
Elevation 109 = 1603
Station File 109 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-120.78125

Station Name 110 = data_48.59375_-120.84375
North Coordinate 110 = 5384541.820000
East Coordinate 110 = 658987.594000
Elevation 110 = 1634
Station File 110 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-120.84375

Station Name 111 = data_48.59375_-120.90625
North Coordinate 111 = 5384413.524000
East Coordinate 111 = 654379.515700
Elevation 111 = 1743
Station File 111 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-120.90625

Station Name 112 = data_48.59375_-120.96875
North Coordinate 112 = 5384289.004000
East Coordinate 112 = 649771.414700
Elevation 112 = 1626
Station File 112 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-120.96875

Station Name 113 = data_48.59375_-121.03125
North Coordinate 113 = 5384168.259000
East Coordinate 113 = 645163.291600
Elevation 113 = 1093
Station File 113 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.03125

Station Name 114 = data_48.59375_-121.09375
North Coordinate 114 = 5384051.291000
East Coordinate 114 = 640555.147200
Elevation 114 = 1605
Station File 114 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.09375

Station Name 115 = data_48.59375_-121.15625
North Coordinate 115 = 5383938.098000
East Coordinate 115 = 635946.982000
Elevation 115 = 1564
Station File 115 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.15625

Station Name 116 = data_48.59375_-121.21875
North Coordinate 116 = 5383828.680000
East Coordinate 116 = 631338.796800
Elevation 116 = 1320
Station File 116 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.21875

Station Name 117 = data_48.59375_-121.28125
North Coordinate 117 = 5383723.037000
East Coordinate 117 = 626730.592300
Elevation 117 = 1569
Station File 117 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.28125

Station Name 118 = data_48.59375_-121.34375
North Coordinate 118 = 5383621.169000
East Coordinate 118 = 622122.369000
Elevation 118 = 846
Station File 118 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.34375

Station Name 119 = data_48.59375_-121.40625
North Coordinate 119 = 5383523.076000
East Coordinate 119 = 617514.127800
Elevation 119 = 427
Station File 119 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.40625

Station Name 120 = data_48.59375_-121.46875
North Coordinate 120 = 5383428.756000
East Coordinate 120 = 612905.869300
Elevation 120 = 984
Station File 120 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.46875

Station Name 121 = data_48.59375_-121.53125
North Coordinate 121 = 5383338.211000
East Coordinate 121 = 608297.594100
Elevation 121 = 1113
Station File 121 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.53125

Station Name 122 = data_48.59375_-121.59375
North Coordinate 122 = 5383251.441000
East Coordinate 122 = 603689.303000
Elevation 122 = 1053
Station File 122 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.59375

Station Name 123 = data_48.59375_-121.65625
North Coordinate 123 = 5383168.443000
East Coordinate 123 = 599080.996600
Elevation 123 = 976
Station File 123 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.65625

Station Name 124 = data_48.59375_-121.71875
North Coordinate 124 = 5383089.220000
East Coordinate 124 = 594472.675600
Elevation 124 = 305
Station File 124 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.71875

Station Name 125 = data_48.59375_-121.78125
North Coordinate 125 = 5383013.770000
East Coordinate 125 = 589864.340600
Elevation 125 = 503
Station File 125 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.78125

Station Name 126 = data_48.59375_-121.84375
North Coordinate 126 = 5382942.094000
East Coordinate 126 = 585255.992500
Elevation 126 = 887
Station File 126 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.59375_-121.84375

Station Name 127 = data_48.53125_-120.71875
North Coordinate 127 = 5377863.122000
East Coordinate 127 = 668411.137000
Elevation 127 = 1805
Station File 127 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-120.71875

Station Name 128 = data_48.53125_-120.78125
North Coordinate 128 = 5377727.236000
East Coordinate 128 = 663797.417800
Elevation 128 = 1682
Station File 128 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-120.78125

Station Name 129 = data_48.53125_-120.90625
North Coordinate 129 = 5377466.797000
East Coordinate 129 = 654569.908900
Elevation 129 = 1690
Station File 129 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-120.90625

Station Name 130 = data_48.53125_-120.96875
North Coordinate 130 = 5377342.243000
East Coordinate 130 = 649956.120500
Elevation 130 = 1939
Station File 130 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-120.96875

Station Name 131 = data_48.53125_-121.03125
North Coordinate 131 = 5377221.466000
East Coordinate 131 = 645342.310500
Elevation 131 = 1588
Station File 131 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.03125

Station Name 132 = data_48.53125_-121.09375
North Coordinate 132 = 5377104.466000
East Coordinate 132 = 640728.479400
Elevation 132 = 1853
Station File 132 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.09375

Station Name 133 = data_48.53125_-121.15625
North Coordinate 133 = 5376991.242000
East Coordinate 133 = 636114.627900
Elevation 133 = 1730
Station File 133 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.15625

Station Name 134 = data_48.53125_-121.21875
North Coordinate 134 = 5376881.795000
East Coordinate 134 = 631500.756800
Elevation 134 = 1166
Station File 134 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.21875

Station Name 135 = data_48.53125_-121.28125
North Coordinate 135 = 5376776.123000
East Coordinate 135 = 626886.866600
Elevation 135 = 794
Station File 135 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.28125

Station Name 136 = data_48.53125_-121.34375
North Coordinate 136 = 5376674.228000
East Coordinate 136 = 622272.958000
Elevation 136 = 832
Station File 136 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.34375

Station Name 137 = data_48.53125_-121.40625
North Coordinate 137 = 5376576.108000
East Coordinate 137 = 617659.031700
Elevation 137 = 354
Station File 137 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.40625

Station Name 138 = data_48.53125_-121.46875
North Coordinate 138 = 5376481.763000
East Coordinate 138 = 613045.088400
Elevation 138 = 378
Station File 138 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.46875

Station Name 139 = data_48.53125_-121.53125
North Coordinate 139 = 5376391.194000
East Coordinate 139 = 608431.128800
Elevation 139 = 873
Station File 139 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.53125

Station Name 140 = data_48.53125_-121.59375
North Coordinate 140 = 5376304.400000
East Coordinate 140 = 603817.153400
Elevation 140 = 1113
Station File 140 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.59375

Station Name 141 = data_48.53125_-121.65625
North Coordinate 141 = 5376221.380000
East Coordinate 141 = 599203.163000
Elevation 141 = 554
Station File 141 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.65625

Station Name 142 = data_48.53125_-121.71875
North Coordinate 142 = 5376142.135000
East Coordinate 142 = 594589.158200
Elevation 142 = 230
Station File 142 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.71875

Station Name 143 = data_48.53125_-121.78125
North Coordinate 143 = 5376066.665000
East Coordinate 143 = 589975.139800
Elevation 143 = 159
Station File 143 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.78125

Station Name 144 = data_48.53125_-121.84375
North Coordinate 144 = 5375994.969000
East Coordinate 144 = 585361.108300
Elevation 144 = 121
Station File 144 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.84375

Station Name 145 = data_48.53125_-121.90625
North Coordinate 145 = 5375927.048000
East Coordinate 145 = 580747.064500
Elevation 145 = 129
Station File 145 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.90625

Station Name 146 = data_48.53125_-121.96875
North Coordinate 146 = 5375862.901000
East Coordinate 146 = 576133.009000
Elevation 146 = 160
Station File 146 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-121.96875

Station Name 147 = data_48.53125_-122.03125
North Coordinate 147 = 5375802.527000
East Coordinate 147 = 571518.942400
Elevation 147 = 78
Station File 147 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-122.03125

Station Name 148 = data_48.53125_-122.09375
North Coordinate 148 = 5375745.928000
East Coordinate 148 = 566904.865500
Elevation 148 = 162
Station File 148 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-122.09375

Station Name 149 = data_48.53125_-122.15625
North Coordinate 149 = 5375693.102000
East Coordinate 149 = 562290.779000
Elevation 149 = 137
Station File 149 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-122.15625

Station Name 150 = data_48.53125_-122.21875
North Coordinate 150 = 5375644.050000
East Coordinate 150 = 557676.683400
Elevation 150 = 61
Station File 150 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.53125_-122.21875

Station Name 151 = data_48.46875_-121.09375
North Coordinate 151 = 5370157.708000
East Coordinate 151 = 640901.642900
Elevation 151 = 1430
Station File 151 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.09375

Station Name 152 = data_48.46875_-121.15625
North Coordinate 152 = 5370044.455000
East Coordinate 152 = 636282.110700
Elevation 152 = 1037
Station File 152 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.15625

Station Name 153 = data_48.46875_-121.21875
North Coordinate 153 = 5369934.978000
East Coordinate 153 = 631662.559100
Elevation 153 = 919
Station File 153 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.21875

Station Name 154 = data_48.46875_-121.28125
North Coordinate 154 = 5369829.279000
East Coordinate 154 = 627042.988700
Elevation 154 = 1257
Station File 154 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.28125

Station Name 155 = data_48.46875_-121.34375
North Coordinate 155 = 5369727.356000
East Coordinate 155 = 622423.400400
Elevation 155 = 1388
Station File 155 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.34375

Station Name 156 = data_48.46875_-121.40625
North Coordinate 156 = 5369629.210000
East Coordinate 156 = 617803.794600
Elevation 156 = 1114
Station File 156 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.40625

Station Name 157 = data_48.46875_-121.46875
North Coordinate 157 = 5369534.840000
East Coordinate 157 = 613184.172000
Elevation 157 = 646
Station File 157 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.46875

Station Name 158 = data_48.46875_-121.53125
North Coordinate 158 = 5369444.247000
East Coordinate 158 = 608564.533400
Elevation 158 = 321
Station File 158 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.53125

Station Name 159 = data_48.46875_-121.59375
North Coordinate 159 = 5369357.430000
East Coordinate 159 = 603944.879300
Elevation 159 = 164
Station File 159 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.59375

Station Name 160 = data_48.46875_-121.65625
North Coordinate 160 = 5369274.388000
East Coordinate 160 = 599325.210400
Elevation 160 = 369
Station File 160 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.65625

Station Name 161 = data_48.46875_-121.71875
North Coordinate 161 = 5369195.123000
East Coordinate 161 = 594705.527500
Elevation 161 = 515
Station File 161 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.71875

Station Name 162 = data_48.46875_-121.78125
North Coordinate 162 = 5369119.632000
East Coordinate 162 = 590085.831000
Elevation 162 = 830
Station File 162 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.78125

Station Name 163 = data_48.46875_-121.84375
North Coordinate 163 = 5369047.918000
East Coordinate 163 = 585466.121800
Elevation 163 = 834
Station File 163 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.84375

Station Name 164 = data_48.46875_-121.90625
North Coordinate 164 = 5368979.978000
East Coordinate 164 = 580846.400400
Elevation 164 = 988
Station File 164 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.90625

Station Name 165 = data_48.46875_-121.96875
North Coordinate 165 = 5368915.814000
East Coordinate 165 = 576226.667500
Elevation 165 = 946
Station File 165 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-121.96875

Station Name 166 = data_48.46875_-122.03125
North Coordinate 166 = 5368855.425000
East Coordinate 166 = 571606.923800
Elevation 166 = 421
Station File 166 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-122.03125

Station Name 167 = data_48.46875_-122.09375
North Coordinate 167 = 5368798.810000
East Coordinate 167 = 566987.169800
Elevation 167 = 521
Station File 167 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-122.09375

Station Name 168 = data_48.46875_-122.15625
North Coordinate 168 = 5368745.971000
East Coordinate 168 = 562367.406400
Elevation 168 = 394
Station File 168 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-122.15625

Station Name 169 = data_48.46875_-122.21875
North Coordinate 169 = 5368696.906000
East Coordinate 169 = 557747.634100
Elevation 169 = 90
Station File 169 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-122.21875

Station Name 170 = data_48.46875_-122.28125
North Coordinate 170 = 5368651.616000
East Coordinate 170 = 553127.853600
Elevation 170 = 16
Station File 170 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.46875_-122.28125

Station Name 171 = data_48.40625_-121.09375
North Coordinate 171 = 5363211.018000
East Coordinate 171 = 641074.637500
Elevation 171 = 1535
Station File 171 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.09375

Station Name 172 = data_48.40625_-121.15625
North Coordinate 172 = 5363097.735000
East Coordinate 172 = 636449.430100
Elevation 172 = 1518
Station File 172 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.15625

Station Name 173 = data_48.40625_-121.21875
North Coordinate 173 = 5362988.230000
East Coordinate 173 = 631824.203500
Elevation 173 = 1218
Station File 173 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.21875

Station Name 174 = data_48.40625_-121.28125
North Coordinate 174 = 5362882.503000
East Coordinate 174 = 627198.958600
Elevation 174 = 1661
Station File 174 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.28125

Station Name 175 = data_48.40625_-121.34375
North Coordinate 175 = 5362780.554000
East Coordinate 175 = 622573.695900
Elevation 175 = 1324
Station File 175 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.34375

Station Name 176 = data_48.40625_-121.40625
North Coordinate 176 = 5362682.382000
East Coordinate 176 = 617948.416100
Elevation 176 = 1097
Station File 176 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.40625

Station Name 177 = data_48.40625_-121.46875
North Coordinate 177 = 5362587.988000
East Coordinate 177 = 613323.119900
Elevation 177 = 1195
Station File 177 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.46875

Station Name 178 = data_48.40625_-121.53125
North Coordinate 178 = 5362497.371000
East Coordinate 178 = 608697.807800
Elevation 178 = 647
Station File 178 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.53125

Station Name 179 = data_48.40625_-121.59375
North Coordinate 179 = 5362410.531000
East Coordinate 179 = 604072.480600
Elevation 179 = 406
Station File 179 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.59375

Station Name 180 = data_48.40625_-121.78125
North Coordinate 180 = 5362172.672000
East Coordinate 180 = 590196.414200
Elevation 180 = 917
Station File 180 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.78125

Station Name 181 = data_48.40625_-121.96875
North Coordinate 181 = 5361968.801000
East Coordinate 181 = 576320.234600
Elevation 181 = 756
Station File 181 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-121.96875

Station Name 182 = data_48.40625_-122.03125
North Coordinate 182 = 5361908.396000
East Coordinate 182 = 571694.819200
Elevation 182 = 828
Station File 182 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-122.03125

Station Name 183 = data_48.40625_-122.09375
North Coordinate 183 = 5361851.767000
East Coordinate 183 = 567069.393800
Elevation 183 = 962
Station File 183 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-122.09375

Station Name 184 = data_48.40625_-122.15625
North Coordinate 184 = 5361798.913000
East Coordinate 184 = 562443.959000
Elevation 184 = 596
Station File 184 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-122.15625

Station Name 185 = data_48.40625_-122.21875
North Coordinate 185 = 5361749.836000
East Coordinate 185 = 557818.515600
Elevation 185 = 80
Station File 185 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.40625_-122.21875

Station Name 186 = data_48.34375_-121.09375
North Coordinate 186 = 5356264.396000
East Coordinate 186 = 641247.463100
Elevation 186 = 1676
Station File 186 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.34375_-121.09375

Station Name 187 = data_48.34375_-121.15625
North Coordinate 187 = 5356151.083000
East Coordinate 187 = 636616.585900
Elevation 187 = 1343
Station File 187 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.34375_-121.15625

Station Name 188 = data_48.34375_-121.21875
North Coordinate 188 = 5356041.550000
East Coordinate 188 = 631985.690000
Elevation 188 = 1567
Station File 188 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.34375_-121.21875

Station Name 189 = data_48.34375_-121.28125
North Coordinate 189 = 5355935.796000
East Coordinate 189 = 627354.776000
Elevation 189 = 1205
Station File 189 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.34375_-121.28125

Station Name 190 = data_48.34375_-121.34375
North Coordinate 190 = 5355833.821000
East Coordinate 190 = 622723.844600
Elevation 190 = 1377
Station File 190 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.34375_-121.34375

Station Name 191 = data_48.34375_-121.40625
North Coordinate 191 = 5355735.625000
East Coordinate 191 = 618092.896300
Elevation 191 = 858
Station File 191 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.34375_-121.40625

Station Name 192 = data_48.34375_-121.46875
North Coordinate 192 = 5355641.206000
East Coordinate 192 = 613461.931900
Elevation 192 = 452
Station File 192 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.34375_-121.46875

Station Name 193 = data_48.34375_-121.53125
North Coordinate 193 = 5355550.566000
East Coordinate 193 = 608830.951900
Elevation 193 = 164
Station File 193 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.34375_-121.53125

Station Name 194 = data_48.34375_-122.15625
North Coordinate 194 = 5354851.930000
East Coordinate 194 = 562520.436800
Elevation 194 = 286
Station File 194 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.34375_-122.15625

Station Name 195 = data_48.34375_-122.21875
North Coordinate 195 = 5354802.840000
East Coordinate 195 = 557889.327700
Elevation 195 = 179
Station File 195 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.34375_-122.21875

Station Name 196 = data_48.28125_-121.03125
North Coordinate 196 = 5349434.962000
East Coordinate 196 = 646056.641600
Elevation 196 = 1768
Station File 196 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.28125_-121.03125

Station Name 197 = data_48.28125_-121.09375
North Coordinate 197 = 5349317.841000
East Coordinate 197 = 641420.119300
Elevation 197 = 1404
Station File 197 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.28125_-121.09375

Station Name 198 = data_48.28125_-121.15625
North Coordinate 198 = 5349204.500000
East Coordinate 198 = 636783.577900
Elevation 198 = 1346
Station File 198 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.28125_-121.15625

Station Name 199 = data_48.28125_-121.21875
North Coordinate 199 = 5349094.939000
East Coordinate 199 = 632147.018200
Elevation 199 = 1087
Station File 199 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.28125_-121.21875

Station Name 200 = data_48.28125_-121.28125
North Coordinate 200 = 5348989.159000
East Coordinate 200 = 627510.440700
Elevation 200 = 1024
Station File 200 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.28125_-121.28125

Station Name 201 = data_48.28125_-121.34375
North Coordinate 201 = 5348887.158000
East Coordinate 201 = 622873.846000
Elevation 201 = 856
Station File 201 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.28125_-121.34375

Station Name 202 = data_48.28125_-121.40625
North Coordinate 202 = 5348788.937000
East Coordinate 202 = 618237.234900
Elevation 202 = 557
Station File 202 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.28125_-121.40625

Station Name 203 = data_48.28125_-121.46875
North Coordinate 203 = 5348694.495000
East Coordinate 203 = 613600.607800
Elevation 203 = 996
Station File 203 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.28125_-121.46875

Station Name 204 = data_48.28125_-121.53125
North Coordinate 204 = 5348603.832000
East Coordinate 204 = 608963.965500
Elevation 204 = 343
Station File 204 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.28125_-121.53125

Station Name 205 = data_48.28125_-121.59375
North Coordinate 205 = 5348516.948000
East Coordinate 205 = 604327.308500
Elevation 205 = 339
Station File 205 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.28125_-121.59375

Station Name 206 = data_48.21875_-120.96875
North Coordinate 206 = 5342609.434000
East Coordinate 206 = 650876.949300
Elevation 206 = 1705
Station File 206 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-120.96875

Station Name 207 = data_48.21875_-121.03125
North Coordinate 207 = 5342488.503000
East Coordinate 207 = 646234.787300
Elevation 207 = 1509
Station File 207 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-121.03125

Station Name 208 = data_48.21875_-121.09375
North Coordinate 208 = 5342371.353000
East Coordinate 208 = 641592.606000
Elevation 208 = 1405
Station File 208 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-121.09375

Station Name 209 = data_48.21875_-121.15625
North Coordinate 209 = 5342257.984000
East Coordinate 209 = 636950.406000
Elevation 209 = 1147
Station File 209 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-121.15625

Station Name 210 = data_48.21875_-121.21875
North Coordinate 210 = 5342148.397000
East Coordinate 210 = 632308.188000
Elevation 210 = 1350
Station File 210 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-121.21875

Station Name 211 = data_48.21875_-121.28125
North Coordinate 211 = 5342042.590000
East Coordinate 211 = 627665.952500
Elevation 211 = 1171
Station File 211 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-121.28125

Station Name 212 = data_48.21875_-121.34375
North Coordinate 212 = 5341940.564000
East Coordinate 212 = 623023.700200
Elevation 212 = 1242
Station File 212 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-121.34375

Station Name 213 = data_48.21875_-121.40625
North Coordinate 213 = 5341842.319000
East Coordinate 213 = 618381.431700
Elevation 213 = 1112
Station File 213 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-121.40625

Station Name 214 = data_48.21875_-121.46875
North Coordinate 214 = 5341747.854000
East Coordinate 214 = 613739.147600
Elevation 214 = 926
Station File 214 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-121.46875

Station Name 215 = data_48.21875_-121.53125
North Coordinate 215 = 5341657.169000
East Coordinate 215 = 609096.848400
Elevation 215 = 613
Station File 215 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-121.53125

Station Name 216 = data_48.21875_-121.59375
North Coordinate 216 = 5341570.264000
East Coordinate 216 = 604454.534900
Elevation 216 = 609
Station File 216 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.21875_-121.59375

Station Name 217 = data_48.15625_-120.96875
North Coordinate 217 = 5335663.072000
East Coordinate 217 = 651060.573500
Elevation 217 = 1732
Station File 217 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-120.96875

Station Name 218 = data_48.15625_-121.03125
North Coordinate 218 = 5335542.111000
East Coordinate 218 = 646412.757700
Elevation 218 = 1273
Station File 218 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.03125

Station Name 219 = data_48.15625_-121.09375
North Coordinate 219 = 5335424.933000
East Coordinate 219 = 641764.923000
Elevation 219 = 1736
Station File 219 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.09375

Station Name 220 = data_48.15625_-121.15625
North Coordinate 220 = 5335311.537000
East Coordinate 220 = 637117.070000
Elevation 220 = 1628
Station File 220 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.15625

Station Name 221 = data_48.15625_-121.21875
North Coordinate 221 = 5335201.923000
East Coordinate 221 = 632469.199200
Elevation 221 = 1347
Station File 221 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.21875

Station Name 222 = data_48.15625_-121.28125
North Coordinate 222 = 5335096.091000
East Coordinate 222 = 627821.311300
Elevation 222 = 1148
Station File 222 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.28125

Station Name 223 = data_48.15625_-121.34375
North Coordinate 223 = 5334994.040000
East Coordinate 223 = 623173.406900
Elevation 223 = 1094
Station File 223 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.34375

Station Name 224 = data_48.15625_-121.40625
North Coordinate 224 = 5334895.771000
East Coordinate 224 = 618525.486600
Elevation 224 = 818
Station File 224 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.40625

Station Name 225 = data_48.15625_-121.46875
North Coordinate 225 = 5334801.283000
East Coordinate 225 = 613877.550900
Elevation 225 = 707
Station File 225 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.46875

Station Name 226 = data_48.15625_-121.53125
North Coordinate 226 = 5334710.577000
East Coordinate 226 = 609229.600600
Elevation 226 = 1014
Station File 226 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.53125

Station Name 227 = data_48.15625_-121.59375
North Coordinate 227 = 5334623.650000
East Coordinate 227 = 604581.636100
Elevation 227 = 912
Station File 227 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.59375

Station Name 228 = data_48.15625_-121.65625
North Coordinate 228 = 5334540.505000
East Coordinate 228 = 599933.658100
Elevation 228 = 1056
Station File 228 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.15625_-121.65625

Station Name 229 = data_48.09375_-121.03125
North Coordinate 229 = 5328595.786000
East Coordinate 229 = 646590.552700
Elevation 229 = 1719
Station File 229 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.09375_-121.03125

Station Name 230 = data_48.09375_-121.09375
North Coordinate 230 = 5328478.580000
East Coordinate 230 = 641937.070100
Elevation 230 = 2216
Station File 230 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.09375_-121.09375

Station Name 231 = data_48.09375_-121.15625
North Coordinate 231 = 5328365.158000
East Coordinate 231 = 637283.569500
Elevation 231 = 1700
Station File 231 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.09375_-121.15625

Station Name 232 = data_48.09375_-121.21875
North Coordinate 232 = 5328255.518000
East Coordinate 232 = 632630.051600
Elevation 232 = 1594
Station File 232 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.09375_-121.21875

Station Name 233 = data_48.09375_-121.28125
North Coordinate 233 = 5328149.661000
East Coordinate 233 = 627976.516900
Elevation 233 = 1274
Station File 233 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.09375_-121.28125

Station Name 234 = data_48.09375_-121.34375
North Coordinate 234 = 5328047.586000
East Coordinate 234 = 623322.965900
Elevation 234 = 1120
Station File 234 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.09375_-121.34375

Station Name 235 = data_48.09375_-121.40625
North Coordinate 235 = 5327949.294000
East Coordinate 235 = 618669.399400
Elevation 235 = 723
Station File 235 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.09375_-121.40625

Station Name 236 = data_48.09375_-121.46875
North Coordinate 236 = 5327854.784000
East Coordinate 236 = 614015.817800
Elevation 236 = 1176
Station File 236 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.09375_-121.46875

Station Name 237 = data_48.03125_-121.15625
North Coordinate 237 = 5321418.846000
East Coordinate 237 = 637449.904600
Elevation 237 = 1585
Station File 237 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.03125_-121.15625

Station Name 238 = data_48.03125_-121.21875
North Coordinate 238 = 5321309.181000
East Coordinate 238 = 632790.745000
Elevation 238 = 1382
Station File 238 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.03125_-121.21875

Station Name 239 = data_48.03125_-121.28125
North Coordinate 239 = 5321203.300000
East Coordinate 239 = 628131.569000
Elevation 239 = 1035
Station File 239 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.03125_-121.28125

Station Name 240 = data_48.03125_-121.34375
North Coordinate 240 = 5321101.201000
East Coordinate 240 = 623472.377100
Elevation 240 = 1373
Station File 240 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.03125_-121.34375

Station Name 241 = data_48.03125_-121.40625
North Coordinate 241 = 5321002.886000
East Coordinate 241 = 618813.169900
Elevation 241 = 1188
Station File 241 = /civil/shared/ecohydrology/Skagit/forcings/DHSVMinput_VICoutputs_biascorrWRF_global/data_48.03125_-121.40625


################ MM5 ###########################################################
# The following block only needs to be filled out if MM5 = TRUE.  In that case
# This is the ONLY block that needs to be filled out

MM5 Start              =                  # Start of MM5 file (MM/DD/YYYY-HH),
MM5 Rows               =
MM5 Cols               =
MM5 Extreme North      =
MM5 Extreme West       = 
MM5 DY                 =

# MM5 met files
MM5 Temperature File   = 
MM5 Humidity File      = 
MM5 Wind Speed File    = 
MM5 Shortwave File     = 
MM5 Longwave File      = 
MM5 Pressure File      = 
MM5 Precipitation File = 
MM5 Terrain File       =
MM5 Temp Lapse File    =

# For each soil layer make a key-entry pair as below (n = 1, ..,
# Number of Soil Layers)

MM5 Soil Temperature File 0 = 
MM5 Soil Temperature File 1 = 
MM5 Soil Temperature File 2 = 

############### RADAR ##########################################################
# The following block only needs to be filled out if Precipitation Source = 
# RADAR.  
Radar Start            =
Radar File             =
Radar Extreme North    =
Radar Extreme West     =
Radar Number of Rows   =
Radar Number of Columns = 
Radar Grid Spacing     =

################ Wind ##########################################################
# The following block only needs to be filled out if Wind Source = MODEL
Number of Wind Maps    =
Wind File Basename     =
Wind Map Met Stations  =

################ Precipitation lapse rate ######################################
# The following block only needs to be filled out if Precipitation lapse rate 
# = MAP
Precipitation lapse rate =

################################################################################
# SOILS INFORMATION SECTION
################################################################################
[SOILS]                                   # Soil information
Soil Map File	     = arc/skagit150_soilclasses.bin   ##skagit.soil.bin
Soil Depth File      = arc/skagit.soildepth.bin #skagit.soildepth.bin ##skagit150_soildepth_veg42_1min.bin ###skagit.soildepth.bin

Number of Soil Types = 244   

Soil Description 1 = 1700202
Lateral Conductivity 1 = 0.00026590
Exponential Decrease 1 = 0.87270411
Depth Threshold 1 = 0.0500000000
Maximum Infiltration 1 = 0.00026590
Capillary Drive 1 = 0.26251171
Surface Albedo 1 = 0.19
Number of Soil Layers 1 = 3
Porosity 1 = 0.5659 0.5659 0.5659
Pore Size Distribution 1 = 0.57355228 0.57355228 0.57355228
Bubbling Pressure 1 = 0.08200502 0.08200502 0.08200502
Field Capacity 1 = 0.0753 0.0753 0.0753
Wilting Point 1 =  0.0264 0.0264 0.0264
Bulk Density 1 = 1150.42 1150.42 1150.42
Vertical Conductivity 1 = 0.00005365 0.00005365 0.00005365
Thermal Conductivity 1 = 0.7800 0.7800 0.7800
Thermal Capacity 1 = 1080033 1080033 1080033
Mannings n 1 = 0.0300

Soil Description 2 = 2247147
Lateral Conductivity 2 = 0.00031000
Exponential Decrease 2 = 0.78572105
Depth Threshold 2 = 0.0500000000
Maximum Infiltration 2 = 0.00031000
Capillary Drive 2 = 0.30658140
Surface Albedo 2 = 0.18
Number of Soil Layers 2 = 3
Porosity 2 = 0.5590 0.5590 0.5590
Pore Size Distribution 2 = 0.57240406 0.57240406 0.57240406
Bubbling Pressure 2 = 0.08275224 0.08275224 0.08275224
Field Capacity 2 = 0.1005 0.1005 0.1005
Wilting Point 2 =  0.0478 0.0478 0.0478
Bulk Density 2 = 1168.67 1168.67 1168.67
Vertical Conductivity 2 = 0.00006184 0.00006184 0.00006184
Thermal Conductivity 2 = 0.7787 0.7787 0.7787
Thermal Capacity 2 = 1085015 1085015 1085015
Mannings n 2 = 0.0300

Soil Description 3 = 2247646
Lateral Conductivity 3 = 0.00003590
Exponential Decrease 3 = 1.58189888
Depth Threshold 3 = 0.0500000000
Maximum Infiltration 3 = 0.00015030
Capillary Drive 3 = 0.43334842
Surface Albedo 3 = 0.15
Number of Soil Layers 3 = 3
Porosity 3 = 0.6698 0.6698 0.6698
Pore Size Distribution 3 = 0.56848111 0.56848111 0.56848111
Bubbling Pressure 3 = 0.08530514 0.08530514 0.08530514
Field Capacity 3 = 0.2927 0.2927 0.2927
Wilting Point 3 =  0.1214 0.1214 0.1214
Bulk Density 3 = 874.95 874.95 874.95
Vertical Conductivity 3 = 0.00003440 0.00003440 0.00003440
Thermal Conductivity 3 = 0.7745 0.7745 0.7745
Thermal Capacity 3 = 1102034 1102034 1102034
Mannings n 3 = 0.0300

Soil Description 4 = 2409820
Lateral Conductivity 4 = 0.00029450
Exponential Decrease 4 = 0.77914739
Depth Threshold 4 = 0.0500000000
Maximum Infiltration 4 = 0.00029450
Capillary Drive 4 = 0.28077586
Surface Albedo 4 = 0.18
Number of Soil Layers 4 = 3
Porosity 4 = 0.5858 0.5858 0.5858
Pore Size Distribution 4 = 0.57538636 0.57538636 0.57538636
Bubbling Pressure 4 = 0.08081148 0.08081148 0.08081148
Field Capacity 4 = 0.0891 0.0891 0.0891
Wilting Point 4 =  0.0331 0.0331 0.0331
Bulk Density 4 = 1097.76 1097.76 1097.76
Vertical Conductivity 4 = 0.00005963 0.00005963 0.00005963
Thermal Conductivity 4 = 0.7820 0.7820 0.7820
Thermal Capacity 4 = 1072077 1072077 1072077
Mannings n 4 = 0.0300

Soil Description 5 = 2409823
Lateral Conductivity 5 = 0.00018010
Exponential Decrease 5 = 0.51482225
Depth Threshold 5 = 0.0500000000
Maximum Infiltration 5 = 0.00026590
Capillary Drive 5 = 0.30931861
Surface Albedo 5 = 0.21
Number of Soil Layers 5 = 3
Porosity 5 = 0.6589 0.6589 0.6589
Pore Size Distribution 5 = 0.57501974 0.57501974 0.57501974
Bubbling Pressure 5 = 0.08105006 0.08105006 0.08105006
Field Capacity 5 = 0.1037 0.1037 0.1037
Wilting Point 5 =  0.0420 0.0420 0.0420
Bulk Density 5 = 903.90 903.90 903.90
Vertical Conductivity 5 = 0.00010928 0.00010928 0.00010928
Thermal Conductivity 5 = 0.7816 0.7816 0.7816
Thermal Capacity 5 = 1073667 1073667 1073667
Mannings n 5 = 0.0300

Soil Description 6 = 2409826
Lateral Conductivity 6 = 0.00029450
Exponential Decrease 6 = 0.96907976
Depth Threshold 6 = 0.0500000000
Maximum Infiltration 6 = 0.00029450
Capillary Drive 6 = 0.28526412
Surface Albedo 6 = 0.21
Number of Soil Layers 6 = 3
Porosity 6 = 0.6081 0.6081 0.6081
Pore Size Distribution 6 = 0.56979940 0.56979940 0.56979940
Bubbling Pressure 6 = 0.08444724 0.08444724 0.08444724
Field Capacity 6 = 0.0853 0.0853 0.0853
Wilting Point 6 =  0.0436 0.0436 0.0436
Bulk Density 6 = 1038.52 1038.52 1038.52
Vertical Conductivity 6 = 0.00005058 0.00005058 0.00005058
Thermal Conductivity 6 = 0.7759 0.7759 0.7759
Thermal Capacity 6 = 1096315 1096315 1096315
Mannings n 6 = 0.0300

Soil Description 7 = 2409827
Lateral Conductivity 7 = 0.00031000
Exponential Decrease 7 = 0.68285067
Depth Threshold 7 = 0.0500000000
Maximum Infiltration 7 = 0.00031000
Capillary Drive 7 = 0.29486623
Surface Albedo 7 = 0.24
Number of Soil Layers 7 = 3
Porosity 7 = 0.5279 0.5279 0.5279
Pore Size Distribution 7 = 0.57573435 0.57573435 0.57573435
Bubbling Pressure 7 = 0.08058502 0.08058502 0.08058502
Field Capacity 7 = 0.0956 0.0956 0.0956
Wilting Point 7 =  0.0360 0.0360 0.0360
Bulk Density 7 = 1251.02 1251.02 1251.02
Vertical Conductivity 7 = 0.00006746 0.00006746 0.00006746
Thermal Conductivity 7 = 0.7824 0.7824 0.7824
Thermal Capacity 7 = 1070567 1070567 1070567
Mannings n 7 = 0.0300

Soil Description 8 = 2409828
Lateral Conductivity 8 = 0.00029450
Exponential Decrease 8 = 1.68331966
Depth Threshold 8 = 0.0500000000
Maximum Infiltration 8 = 0.00029450
Capillary Drive 8 = 0.38532646
Surface Albedo 8 = 0.21
Number of Soil Layers 8 = 3
Porosity 8 = 0.6453 0.6453 0.6453
Pore Size Distribution 8 = 0.56972594 0.56972594 0.56972594
Bubbling Pressure 8 = 0.08449505 0.08449505 0.08449505
Field Capacity 8 = 0.1181 0.1181 0.1181
Wilting Point 8 =  0.0453 0.0453 0.0453
Bulk Density 8 = 939.85 939.85 939.85
Vertical Conductivity 8 = 0.00004022 0.00004022 0.00004022
Thermal Conductivity 8 = 0.7758 0.7758 0.7758
Thermal Capacity 8 = 1096634 1096634 1096634
Mannings n 8 = 0.0300

Soil Description 9 = 2421681
Lateral Conductivity 9 = 0.00026350
Exponential Decrease 9 = 1.40590418
Depth Threshold 9 = 0.0500000000
Maximum Infiltration 9 = 0.00026350
Capillary Drive 9 = 0.32901845
Surface Albedo 9 = 0.19
Number of Soil Layers 9 = 3
Porosity 9 = 0.7018 0.7018 0.7018
Pore Size Distribution 9 = 0.57502562 0.57502562 0.57502562
Bubbling Pressure 9 = 0.08104624 0.08104624 0.08104624
Field Capacity 9 = 0.1017 0.1017 0.1017
Wilting Point 9 =  0.0407 0.0407 0.0407
Bulk Density 9 = 790.12 790.12 790.12
Vertical Conductivity 9 = 0.00004729 0.00004729 0.00004729
Thermal Conductivity 9 = 0.7816 0.7816 0.7816
Thermal Capacity 9 = 1073642 1073642 1073642
Mannings n 9 = 0.0300

Soil Description 10 = 2421682
Lateral Conductivity 10 = 0.00031000
Exponential Decrease 10 = 0.82753790
Depth Threshold 10 = 0.0500000000
Maximum Infiltration 10 = 0.00031000
Capillary Drive 10 = 0.34497434
Surface Albedo 10 = 0.17
Number of Soil Layers 10 = 3
Porosity 10 = 0.5785 0.5785 0.5785
Pore Size Distribution 10 = 0.57337370 0.57337370 0.57337370
Bubbling Pressure 10 = 0.08212123 0.08212123 0.08212123
Field Capacity 10 = 0.1079 0.1079 0.1079
Wilting Point 10 =  0.0447 0.0447 0.0447
Bulk Density 10 = 1116.90 1116.90 1116.90
Vertical Conductivity 10 = 0.00005930 0.00005930 0.00005930
Thermal Conductivity 10 = 0.7798 0.7798 0.7798
Thermal Capacity 10 = 1080808 1080808 1080808
Mannings n 10 = 0.0300

Soil Description 11 = 2421684
Lateral Conductivity 11 = 0.00028020
Exponential Decrease 11 = 0.50969337
Depth Threshold 11 = 0.0500000000
Maximum Infiltration 11 = 0.00028020
Capillary Drive 11 = 0.37223798
Surface Albedo 11 = 0.22
Number of Soil Layers 11 = 3
Porosity 11 = 0.5253 0.5253 0.5253
Pore Size Distribution 11 = 0.58211306 0.58211306 0.58211306
Bubbling Pressure 11 = 0.07643402 0.07643402 0.07643402
Field Capacity 11 = 0.0974 0.0974 0.0974
Wilting Point 11 =  0.0314 0.0314 0.0314
Bulk Density 11 = 1258.02 1258.02 1258.02
Vertical Conductivity 11 = 0.00008889 0.00008889 0.00008889
Thermal Conductivity 11 = 0.7893 0.7893 0.7893
Thermal Capacity 11 = 1042893 1042893 1042893
Mannings n 11 = 0.0300

Soil Description 12 = 2421688
Lateral Conductivity 12 = 0.00024800
Exponential Decrease 12 = 1.46838030
Depth Threshold 12 = 0.0500000000
Maximum Infiltration 12 = 0.00024800
Capillary Drive 12 = 0.31769547
Surface Albedo 12 = 0.22
Number of Soil Layers 12 = 3
Porosity 12 = 0.7329 0.7329 0.7329
Pore Size Distribution 12 = 0.57634208 0.57634208 0.57634208
Bubbling Pressure 12 = 0.08018953 0.08018953 0.08018953
Field Capacity 12 = 0.0917 0.0917 0.0917
Wilting Point 12 =  0.0340 0.0340 0.0340
Bulk Density 12 = 707.79 707.79 707.79
Vertical Conductivity 12 = 0.00005258 0.00005258 0.00005258
Thermal Conductivity 12 = 0.7830 0.7830 0.7830
Thermal Capacity 12 = 1067930 1067930 1067930
Mannings n 12 = 0.0300

Soil Description 13 = 2421700
Lateral Conductivity 13 = 0.00026590
Exponential Decrease 13 = 0.86323080
Depth Threshold 13 = 0.0500000000
Maximum Infiltration 13 = 0.00029450
Capillary Drive 13 = 0.31245734
Surface Albedo 13 = 0.29
Number of Soil Layers 13 = 3
Porosity 13 = 0.6225 0.6225 0.6225
Pore Size Distribution 13 = 0.56952721 0.56952721 0.56952721
Bubbling Pressure 13 = 0.08462438 0.08462438 0.08462438
Field Capacity 13 = 0.0988 0.0988 0.0988
Wilting Point 13 =  0.0368 0.0368 0.0368
Bulk Density 13 = 1000.25 1000.25 1000.25
Vertical Conductivity 13 = 0.00005527 0.00005527 0.00005527
Thermal Conductivity 13 = 0.7756 0.7756 0.7756
Thermal Capacity 13 = 1097496 1097496 1097496
Mannings n 13 = 0.0300

Soil Description 14 = 2421701
Lateral Conductivity 14 = 0.00016340
Exponential Decrease 14 = 2.97911854
Depth Threshold 14 = 0.0500000000
Maximum Infiltration 14 = 0.00016340
Capillary Drive 14 = 0.34903431
Surface Albedo 14 = 0.24
Number of Soil Layers 14 = 3
Porosity 14 = 0.7147 0.7147 0.7147
Pore Size Distribution 14 = 0.56482804 0.56482804 0.56482804
Bubbling Pressure 14 = 0.08768241 0.08768241 0.08768241
Field Capacity 14 = 0.1152 0.1152 0.1152
Wilting Point 14 =  0.0504 0.0504 0.0504
Bulk Density 14 = 756.04 756.04 756.04
Vertical Conductivity 14 = 0.00003487 0.00003487 0.00003487
Thermal Conductivity 14 = 0.7705 0.7705 0.7705
Thermal Capacity 14 = 1117883 1117883 1117883
Mannings n 14 = 0.0300

Soil Description 15 = 2481015
Lateral Conductivity 15 = 0.00006200
Exponential Decrease 15 = 0.50000000
Depth Threshold 15 = 0.0500000000
Maximum Infiltration 15 = 0.00006200
Capillary Drive 15 = 0.10152227
Surface Albedo 15 = 0.06
Number of Soil Layers 15 = 3
Porosity 15 = 0.9427 0.9427 0.9427
Pore Size Distribution 15 = 0.56464596 0.56464596 0.56464596
Bubbling Pressure 15 = 0.08780089 0.08780089 0.08780089
Field Capacity 15 = 0.0352 0.0352 0.0352
Wilting Point 15 =  0.0151 0.0151 0.0151
Bulk Density 15 = 151.72 151.72 151.72
Vertical Conductivity 15 = 0.00001085 0.00001085 0.00001085
Thermal Conductivity 15 = 0.7703 0.7703 0.7703
Thermal Capacity 15 = 1118673 1118673 1118673
Mannings n 15 = 0.1000

Soil Description 16 = 2481016
Lateral Conductivity 16 = 0.00005471
Exponential Decrease 16 = 0.50000000
Depth Threshold 16 = 0.0500000000
Maximum Infiltration 16 = 0.00005471
Capillary Drive 16 = 0.08017903
Surface Albedo 16 = 0.05
Number of Soil Layers 16 = 3
Porosity 16 = 0.9259 0.9259 0.9259
Pore Size Distribution 16 = 0.55030749 0.55030749 0.55030749
Bubbling Pressure 16 = 0.09713178 0.09713178 0.09713178
Field Capacity 16 = 0.0395 0.0395 0.0395
Wilting Point 16 =  0.0202 0.0202 0.0202
Bulk Density 16 = 196.27 196.27 196.27
Vertical Conductivity 16 = 0.00001082 0.00001082 0.00001082
Thermal Conductivity 16 = 0.7548 0.7548 0.7548
Thermal Capacity 16 = 1180879 1180879 1180879
Mannings n 16 = 0.1000

Soil Description 17 = 2487347
Lateral Conductivity 17 = 0.00019320
Exponential Decrease 17 = 0.96732454
Depth Threshold 17 = 0.0500000000
Maximum Infiltration 17 = 0.00019320
Capillary Drive 17 = 0.30262957
Surface Albedo 17 = 0.27
Number of Soil Layers 17 = 3
Porosity 17 = 0.6598 0.6598 0.6598
Pore Size Distribution 17 = 0.57181878 0.57181878 0.57181878
Bubbling Pressure 17 = 0.08313312 0.08313312 0.08313312
Field Capacity 17 = 0.1000 0.1000 0.1000
Wilting Point 17 =  0.0416 0.0416 0.0416
Bulk Density 17 = 901.41 901.41 901.41
Vertical Conductivity 17 = 0.00005957 0.00005957 0.00005957
Thermal Conductivity 17 = 0.7781 0.7781 0.7781
Thermal Capacity 17 = 1087554 1087554 1087554
Mannings n 17 = 0.0300

Soil Description 18 = 2487354
Lateral Conductivity 18 = 0.00025280
Exponential Decrease 18 = 0.60888177
Depth Threshold 18 = 0.0500000000
Maximum Infiltration 18 = 0.00029570
Capillary Drive 18 = 0.22997395
Surface Albedo 18 = 0.25
Number of Soil Layers 18 = 3
Porosity 18 = 0.5487 0.5487 0.5487
Pore Size Distribution 18 = 0.58023480 0.58023480 0.58023480
Bubbling Pressure 18 = 0.07765631 0.07765631 0.07765631
Field Capacity 18 = 0.0690 0.0690 0.0690
Wilting Point 18 =  0.0262 0.0262 0.0262
Bulk Density 18 = 1196.05 1196.05 1196.05
Vertical Conductivity 18 = 0.00007237 0.00007237 0.00007237
Thermal Conductivity 18 = 0.7872 0.7872 0.7872
Thermal Capacity 18 = 1051042 1051042 1051042
Mannings n 18 = 0.0300

Soil Description 19 = 2487355
Lateral Conductivity 19 = 0.00026710
Exponential Decrease 19 = 0.55704000
Depth Threshold 19 = 0.0500000000
Maximum Infiltration 19 = 0.00031000
Capillary Drive 19 = 0.24091557
Surface Albedo 19 = 0.20
Number of Soil Layers 19 = 3
Porosity 19 = 0.5155 0.5155 0.5155
Pore Size Distribution 19 = 0.57824394 0.57824394 0.57824394
Bubbling Pressure 19 = 0.07895188 0.07895188 0.07895188
Field Capacity 19 = 0.0791 0.0791 0.0791
Wilting Point 19 =  0.0289 0.0289 0.0289
Bulk Density 19 = 1283.98 1283.98 1283.98
Vertical Conductivity 19 = 0.00007837 0.00007837 0.00007837
Thermal Conductivity 19 = 0.7851 0.7851 0.7851
Thermal Capacity 19 = 1059679 1059679 1059679
Mannings n 19 = 0.0300

Soil Description 20 = 2487356
Lateral Conductivity 20 = 0.00028140
Exponential Decrease 20 = 0.77128796
Depth Threshold 20 = 0.0500000000
Maximum Infiltration 20 = 0.00028140
Capillary Drive 20 = 0.25201171
Surface Albedo 20 = 0.27
Number of Soil Layers 20 = 3
Porosity 20 = 0.5082 0.5082 0.5082
Pore Size Distribution 20 = 0.57739889 0.57739889 0.57739889
Bubbling Pressure 20 = 0.07950181 0.07950181 0.07950181
Field Capacity 20 = 0.0813 0.0813 0.0813
Wilting Point 20 =  0.0310 0.0310 0.0310
Bulk Density 20 = 1303.20 1303.20 1303.20
Vertical Conductivity 20 = 0.00005774 0.00005774 0.00005774
Thermal Conductivity 20 = 0.7842 0.7842 0.7842
Thermal Capacity 20 = 1063345 1063345 1063345
Mannings n 20 = 0.0300

Soil Description 21 = 2487358
Lateral Conductivity 21 = 0.00018010
Exponential Decrease 21 = 1.47489500
Depth Threshold 21 = 0.0500000000
Maximum Infiltration 21 = 0.00018010
Capillary Drive 21 = 0.30303333
Surface Albedo 21 = 0.27
Number of Soil Layers 21 = 3
Porosity 21 = 0.4851 0.4851 0.4851
Pore Size Distribution 21 = 0.56751010 0.56751010 0.56751010
Bubbling Pressure 21 = 0.08593703 0.08593703 0.08593703
Field Capacity 21 = 0.1210 0.1210 0.1210
Wilting Point 21 =  0.0442 0.0442 0.0442
Bulk Density 21 = 1364.44 1364.44 1364.44
Vertical Conductivity 21 = 0.00002607 0.00002607 0.00002607
Thermal Conductivity 21 = 0.7734 0.7734 0.7734
Thermal Capacity 21 = 1106247 1106247 1106247
Mannings n 21 = 0.0300

Soil Description 22 = 2487359
Lateral Conductivity 22 = 0.00006570
Exponential Decrease 22 = 0.52382240
Depth Threshold 22 = 0.0500000000
Maximum Infiltration 22 = 0.00023730
Capillary Drive 22 = 0.18904048
Surface Albedo 22 = 0.09
Number of Soil Layers 22 = 3
Porosity 22 = 0.6235 0.6235 0.6235
Pore Size Distribution 22 = 0.57793007 0.57793007 0.57793007
Bubbling Pressure 22 = 0.07915614 0.07915614 0.07915614
Field Capacity 22 = 0.0580 0.0580 0.0580
Wilting Point 22 =  0.0248 0.0248 0.0248
Bulk Density 22 = 997.74 997.74 997.74
Vertical Conductivity 22 = 0.00008986 0.00008986 0.00008986
Thermal Conductivity 22 = 0.7847 0.7847 0.7847
Thermal Capacity 22 = 1061041 1061041 1061041
Mannings n 22 = 0.0300

Soil Description 23 = 2487361
Lateral Conductivity 23 = 0.00029450
Exponential Decrease 23 = 0.97624682
Depth Threshold 23 = 0.0500000000
Maximum Infiltration 23 = 0.00029450
Capillary Drive 23 = 0.32148044
Surface Albedo 23 = 0.23
Number of Soil Layers 23 = 3
Porosity 23 = 0.6041 0.6041 0.6041
Pore Size Distribution 23 = 0.57561949 0.57561949 0.57561949
Bubbling Pressure 23 = 0.08065977 0.08065977 0.08065977
Field Capacity 23 = 0.0895 0.0895 0.0895
Wilting Point 23 =  0.0311 0.0311 0.0311
Bulk Density 23 = 1049.24 1049.24 1049.24
Vertical Conductivity 23 = 0.00005210 0.00005210 0.00005210
Thermal Conductivity 23 = 0.7822 0.7822 0.7822
Thermal Capacity 23 = 1071065 1071065 1071065
Mannings n 23 = 0.0300

Soil Description 24 = 2487365
Lateral Conductivity 24 = 0.00031000
Exponential Decrease 24 = 0.75623377
Depth Threshold 24 = 0.0500000000
Maximum Infiltration 24 = 0.00031000
Capillary Drive 24 = 0.28751640
Surface Albedo 24 = 0.25
Number of Soil Layers 24 = 3
Porosity 24 = 0.5278 0.5278 0.5278
Pore Size Distribution 24 = 0.57236275 0.57236275 0.57236275
Bubbling Pressure 24 = 0.08277912 0.08277912 0.08277912
Field Capacity 24 = 0.0968 0.0968 0.0968
Wilting Point 24 =  0.0374 0.0374 0.0374
Bulk Density 24 = 1251.40 1251.40 1251.40
Vertical Conductivity 24 = 0.00005755 0.00005755 0.00005755
Thermal Conductivity 24 = 0.7787 0.7787 0.7787
Thermal Capacity 24 = 1085194 1085194 1085194
Mannings n 24 = 0.0300

Soil Description 25 = 2487366
Lateral Conductivity 25 = 0.00029450
Exponential Decrease 25 = 0.83877980
Depth Threshold 25 = 0.0500000000
Maximum Infiltration 25 = 0.00029450
Capillary Drive 25 = 0.31221640
Surface Albedo 25 = 0.24
Number of Soil Layers 25 = 3
Porosity 25 = 0.5764 0.5764 0.5764
Pore Size Distribution 25 = 0.57559602 0.57559602 0.57559602
Bubbling Pressure 25 = 0.08067504 0.08067504 0.08067504
Field Capacity 25 = 0.0881 0.0881 0.0881
Wilting Point 25 =  0.0302 0.0302 0.0302
Bulk Density 25 = 1122.67 1122.67 1122.67
Vertical Conductivity 25 = 0.00005343 0.00005343 0.00005343
Thermal Conductivity 25 = 0.7822 0.7822 0.7822
Thermal Capacity 25 = 1071167 1071167 1071167
Mannings n 25 = 0.0300

Soil Description 26 = 2487369
Lateral Conductivity 26 = 0.00012290
Exponential Decrease 26 = 1.17843191
Depth Threshold 26 = 0.0500000000
Maximum Infiltration 26 = 0.00012290
Capillary Drive 26 = 0.35623710
Surface Albedo 26 = 0.18
Number of Soil Layers 26 = 3
Porosity 26 = 0.5106 0.5106 0.5106
Pore Size Distribution 26 = 0.57574436 0.57574436 0.57574436
Bubbling Pressure 26 = 0.08057851 0.08057851 0.08057851
Field Capacity 26 = 0.1147 0.1147 0.1147
Wilting Point 26 =  0.0342 0.0342 0.0342
Bulk Density 26 = 1296.92 1296.92 1296.92
Vertical Conductivity 26 = 0.00004124 0.00004124 0.00004124
Thermal Conductivity 26 = 0.7824 0.7824 0.7824
Thermal Capacity 26 = 1070523 1070523 1070523
Mannings n 26 = 0.0300

Soil Description 27 = 2517811
Lateral Conductivity 27 = 0.00031000
Exponential Decrease 27 = 0.68285067
Depth Threshold 27 = 0.0500000000
Maximum Infiltration 27 = 0.00031000
Capillary Drive 27 = 0.29486623
Surface Albedo 27 = 0.24
Number of Soil Layers 27 = 3
Porosity 27 = 0.5279 0.5279 0.5279
Pore Size Distribution 27 = 0.57573435 0.57573435 0.57573435
Bubbling Pressure 27 = 0.08058502 0.08058502 0.08058502
Field Capacity 27 = 0.0956 0.0956 0.0956
Wilting Point 27 =  0.0360 0.0360 0.0360
Bulk Density 27 = 1251.02 1251.02 1251.02
Vertical Conductivity 27 = 0.00006746 0.00006746 0.00006746
Thermal Conductivity 27 = 0.7824 0.7824 0.7824
Thermal Capacity 27 = 1070567 1070567 1070567
Mannings n 27 = 0.0300

Soil Description 28 = 2517813
Lateral Conductivity 28 = 0.00029450
Exponential Decrease 28 = 0.83877980
Depth Threshold 28 = 0.0500000000
Maximum Infiltration 28 = 0.00029450
Capillary Drive 28 = 0.31221640
Surface Albedo 28 = 0.24
Number of Soil Layers 28 = 3
Porosity 28 = 0.5764 0.5764 0.5764
Pore Size Distribution 28 = 0.57559602 0.57559602 0.57559602
Bubbling Pressure 28 = 0.08067504 0.08067504 0.08067504
Field Capacity 28 = 0.0881 0.0881 0.0881
Wilting Point 28 =  0.0302 0.0302 0.0302
Bulk Density 28 = 1122.67 1122.67 1122.67
Vertical Conductivity 28 = 0.00005343 0.00005343 0.00005343
Thermal Conductivity 28 = 0.7822 0.7822 0.7822
Thermal Capacity 28 = 1071167 1071167 1071167
Mannings n 28 = 0.0300

Soil Description 29 = 2517814
Lateral Conductivity 29 = 0.00031000
Exponential Decrease 29 = 0.75623377
Depth Threshold 29 = 0.0500000000
Maximum Infiltration 29 = 0.00031000
Capillary Drive 29 = 0.28751640
Surface Albedo 29 = 0.25
Number of Soil Layers 29 = 3
Porosity 29 = 0.5278 0.5278 0.5278
Pore Size Distribution 29 = 0.57236275 0.57236275 0.57236275
Bubbling Pressure 29 = 0.08277912 0.08277912 0.08277912
Field Capacity 29 = 0.0968 0.0968 0.0968
Wilting Point 29 =  0.0374 0.0374 0.0374
Bulk Density 29 = 1251.40 1251.40 1251.40
Vertical Conductivity 29 = 0.00005755 0.00005755 0.00005755
Thermal Conductivity 29 = 0.7787 0.7787 0.7787
Thermal Capacity 29 = 1085194 1085194 1085194
Mannings n 29 = 0.0300

Soil Description 30 = 2517815
Lateral Conductivity 30 = 0.00024800
Exponential Decrease 30 = 1.46838030
Depth Threshold 30 = 0.0500000000
Maximum Infiltration 30 = 0.00024800
Capillary Drive 30 = 0.31769547
Surface Albedo 30 = 0.22
Number of Soil Layers 30 = 3
Porosity 30 = 0.7329 0.7329 0.7329
Pore Size Distribution 30 = 0.57634208 0.57634208 0.57634208
Bubbling Pressure 30 = 0.08018953 0.08018953 0.08018953
Field Capacity 30 = 0.0917 0.0917 0.0917
Wilting Point 30 =  0.0340 0.0340 0.0340
Bulk Density 30 = 707.79 707.79 707.79
Vertical Conductivity 30 = 0.00005258 0.00005258 0.00005258
Thermal Conductivity 30 = 0.7830 0.7830 0.7830
Thermal Capacity 30 = 1067930 1067930 1067930
Mannings n 30 = 0.0300

Soil Description 31 = 2517816
Lateral Conductivity 31 = 0.00029450
Exponential Decrease 31 = 0.97624682
Depth Threshold 31 = 0.0500000000
Maximum Infiltration 31 = 0.00029450
Capillary Drive 31 = 0.32148044
Surface Albedo 31 = 0.23
Number of Soil Layers 31 = 3
Porosity 31 = 0.6041 0.6041 0.6041
Pore Size Distribution 31 = 0.57561949 0.57561949 0.57561949
Bubbling Pressure 31 = 0.08065977 0.08065977 0.08065977
Field Capacity 31 = 0.0895 0.0895 0.0895
Wilting Point 31 =  0.0311 0.0311 0.0311
Bulk Density 31 = 1049.24 1049.24 1049.24
Vertical Conductivity 31 = 0.00005210 0.00005210 0.00005210
Thermal Conductivity 31 = 0.7822 0.7822 0.7822
Thermal Capacity 31 = 1071065 1071065 1071065
Mannings n 31 = 0.0300

Soil Description 32 = 2517817
Lateral Conductivity 32 = 0.00026350
Exponential Decrease 32 = 1.40590418
Depth Threshold 32 = 0.0500000000
Maximum Infiltration 32 = 0.00026350
Capillary Drive 32 = 0.32901845
Surface Albedo 32 = 0.19
Number of Soil Layers 32 = 3
Porosity 32 = 0.7018 0.7018 0.7018
Pore Size Distribution 32 = 0.57502562 0.57502562 0.57502562
Bubbling Pressure 32 = 0.08104624 0.08104624 0.08104624
Field Capacity 32 = 0.1017 0.1017 0.1017
Wilting Point 32 =  0.0407 0.0407 0.0407
Bulk Density 32 = 790.12 790.12 790.12
Vertical Conductivity 32 = 0.00004729 0.00004729 0.00004729
Thermal Conductivity 32 = 0.7816 0.7816 0.7816
Thermal Capacity 32 = 1073642 1073642 1073642
Mannings n 32 = 0.0300

Soil Description 33 = 2517818
Lateral Conductivity 33 = 0.00029450
Exponential Decrease 33 = 1.68331966
Depth Threshold 33 = 0.0500000000
Maximum Infiltration 33 = 0.00029450
Capillary Drive 33 = 0.38532646
Surface Albedo 33 = 0.21
Number of Soil Layers 33 = 3
Porosity 33 = 0.6453 0.6453 0.6453
Pore Size Distribution 33 = 0.56972594 0.56972594 0.56972594
Bubbling Pressure 33 = 0.08449505 0.08449505 0.08449505
Field Capacity 33 = 0.1181 0.1181 0.1181
Wilting Point 33 =  0.0453 0.0453 0.0453
Bulk Density 33 = 939.85 939.85 939.85
Vertical Conductivity 33 = 0.00004022 0.00004022 0.00004022
Thermal Conductivity 33 = 0.7758 0.7758 0.7758
Thermal Capacity 33 = 1096634 1096634 1096634
Mannings n 33 = 0.0300

Soil Description 34 = 2517821
Lateral Conductivity 34 = 0.00025280
Exponential Decrease 34 = 0.60888177
Depth Threshold 34 = 0.0500000000
Maximum Infiltration 34 = 0.00029570
Capillary Drive 34 = 0.22997395
Surface Albedo 34 = 0.25
Number of Soil Layers 34 = 3
Porosity 34 = 0.5487 0.5487 0.5487
Pore Size Distribution 34 = 0.58023480 0.58023480 0.58023480
Bubbling Pressure 34 = 0.07765631 0.07765631 0.07765631
Field Capacity 34 = 0.0690 0.0690 0.0690
Wilting Point 34 =  0.0262 0.0262 0.0262
Bulk Density 34 = 1196.05 1196.05 1196.05
Vertical Conductivity 34 = 0.00007237 0.00007237 0.00007237
Thermal Conductivity 34 = 0.7872 0.7872 0.7872
Thermal Capacity 34 = 1051042 1051042 1051042
Mannings n 34 = 0.0300

Soil Description 35 = 2517823
Lateral Conductivity 35 = 0.00026590
Exponential Decrease 35 = 0.87270411
Depth Threshold 35 = 0.0500000000
Maximum Infiltration 35 = 0.00026590
Capillary Drive 35 = 0.26251171
Surface Albedo 35 = 0.19
Number of Soil Layers 35 = 3
Porosity 35 = 0.5659 0.5659 0.5659
Pore Size Distribution 35 = 0.57355228 0.57355228 0.57355228
Bubbling Pressure 35 = 0.08200502 0.08200502 0.08200502
Field Capacity 35 = 0.0753 0.0753 0.0753
Wilting Point 35 =  0.0264 0.0264 0.0264
Bulk Density 35 = 1150.42 1150.42 1150.42
Vertical Conductivity 35 = 0.00005365 0.00005365 0.00005365
Thermal Conductivity 35 = 0.7800 0.7800 0.7800
Thermal Capacity 35 = 1080033 1080033 1080033
Mannings n 35 = 0.0300

Soil Description 36 = 2517824
Lateral Conductivity 36 = 0.00018010
Exponential Decrease 36 = 0.51482225
Depth Threshold 36 = 0.0500000000
Maximum Infiltration 36 = 0.00026590
Capillary Drive 36 = 0.30931861
Surface Albedo 36 = 0.21
Number of Soil Layers 36 = 3
Porosity 36 = 0.6589 0.6589 0.6589
Pore Size Distribution 36 = 0.57501974 0.57501974 0.57501974
Bubbling Pressure 36 = 0.08105006 0.08105006 0.08105006
Field Capacity 36 = 0.1037 0.1037 0.1037
Wilting Point 36 =  0.0420 0.0420 0.0420
Bulk Density 36 = 903.90 903.90 903.90
Vertical Conductivity 36 = 0.00010928 0.00010928 0.00010928
Thermal Conductivity 36 = 0.7816 0.7816 0.7816
Thermal Capacity 36 = 1073667 1073667 1073667
Mannings n 36 = 0.0300

Soil Description 37 = 2517825
Lateral Conductivity 37 = 0.00026590
Exponential Decrease 37 = 0.86323080
Depth Threshold 37 = 0.0500000000
Maximum Infiltration 37 = 0.00029450
Capillary Drive 37 = 0.31245734
Surface Albedo 37 = 0.29
Number of Soil Layers 37 = 3
Porosity 37 = 0.6225 0.6225 0.6225
Pore Size Distribution 37 = 0.56952721 0.56952721 0.56952721
Bubbling Pressure 37 = 0.08462438 0.08462438 0.08462438
Field Capacity 37 = 0.0988 0.0988 0.0988
Wilting Point 37 =  0.0368 0.0368 0.0368
Bulk Density 37 = 1000.25 1000.25 1000.25
Vertical Conductivity 37 = 0.00005527 0.00005527 0.00005527
Thermal Conductivity 37 = 0.7756 0.7756 0.7756
Thermal Capacity 37 = 1097496 1097496 1097496
Mannings n 37 = 0.0300

Soil Description 38 = 2517827
Lateral Conductivity 38 = 0.00016340
Exponential Decrease 38 = 2.97911854
Depth Threshold 38 = 0.0500000000
Maximum Infiltration 38 = 0.00016340
Capillary Drive 38 = 0.34903431
Surface Albedo 38 = 0.24
Number of Soil Layers 38 = 3
Porosity 38 = 0.7147 0.7147 0.7147
Pore Size Distribution 38 = 0.56482804 0.56482804 0.56482804
Bubbling Pressure 38 = 0.08768241 0.08768241 0.08768241
Field Capacity 38 = 0.1152 0.1152 0.1152
Wilting Point 38 =  0.0504 0.0504 0.0504
Bulk Density 38 = 756.04 756.04 756.04
Vertical Conductivity 38 = 0.00003487 0.00003487 0.00003487
Thermal Conductivity 38 = 0.7705 0.7705 0.7705
Thermal Capacity 38 = 1117883 1117883 1117883
Mannings n 38 = 0.0300

Soil Description 39 = 2517828
Lateral Conductivity 39 = 0.00029450
Exponential Decrease 39 = 0.77914739
Depth Threshold 39 = 0.0500000000
Maximum Infiltration 39 = 0.00029450
Capillary Drive 39 = 0.28077586
Surface Albedo 39 = 0.18
Number of Soil Layers 39 = 3
Porosity 39 = 0.5858 0.5858 0.5858
Pore Size Distribution 39 = 0.57538636 0.57538636 0.57538636
Bubbling Pressure 39 = 0.08081148 0.08081148 0.08081148
Field Capacity 39 = 0.0891 0.0891 0.0891
Wilting Point 39 =  0.0331 0.0331 0.0331
Bulk Density 39 = 1097.76 1097.76 1097.76
Vertical Conductivity 39 = 0.00005963 0.00005963 0.00005963
Thermal Conductivity 39 = 0.7820 0.7820 0.7820
Thermal Capacity 39 = 1072077 1072077 1072077
Mannings n 39 = 0.0300

Soil Description 40 = 2517829
Lateral Conductivity 40 = 0.00019320
Exponential Decrease 40 = 0.96732454
Depth Threshold 40 = 0.0500000000
Maximum Infiltration 40 = 0.00019320
Capillary Drive 40 = 0.30262957
Surface Albedo 40 = 0.27
Number of Soil Layers 40 = 3
Porosity 40 = 0.6598 0.6598 0.6598
Pore Size Distribution 40 = 0.57181878 0.57181878 0.57181878
Bubbling Pressure 40 = 0.08313312 0.08313312 0.08313312
Field Capacity 40 = 0.1000 0.1000 0.1000
Wilting Point 40 =  0.0416 0.0416 0.0416
Bulk Density 40 = 901.41 901.41 901.41
Vertical Conductivity 40 = 0.00005957 0.00005957 0.00005957
Thermal Conductivity 40 = 0.7781 0.7781 0.7781
Thermal Capacity 40 = 1087554 1087554 1087554
Mannings n 40 = 0.0300

Soil Description 41 = 2517830
Lateral Conductivity 41 = 0.00006200
Exponential Decrease 41 = 0.50000000
Depth Threshold 41 = 0.0500000000
Maximum Infiltration 41 = 0.00006200
Capillary Drive 41 = 0.10152227
Surface Albedo 41 = 0.06
Number of Soil Layers 41 = 3
Porosity 41 = 0.9427 0.9427 0.9427
Pore Size Distribution 41 = 0.56464596 0.56464596 0.56464596
Bubbling Pressure 41 = 0.08780089 0.08780089 0.08780089
Field Capacity 41 = 0.0352 0.0352 0.0352
Wilting Point 41 =  0.0151 0.0151 0.0151
Bulk Density 41 = 151.72 151.72 151.72
Vertical Conductivity 41 = 0.00001085 0.00001085 0.00001085
Thermal Conductivity 41 = 0.7703 0.7703 0.7703
Thermal Capacity 41 = 1118673 1118673 1118673
Mannings n 41 = 0.1000

Soil Description 42 = 2517831
Lateral Conductivity 42 = 0.00005471
Exponential Decrease 42 = 0.50000000
Depth Threshold 42 = 0.0500000000
Maximum Infiltration 42 = 0.00005471
Capillary Drive 42 = 0.08017903
Surface Albedo 42 = 0.05
Number of Soil Layers 42 = 3
Porosity 42 = 0.9259 0.9259 0.9259
Pore Size Distribution 42 = 0.55030749 0.55030749 0.55030749
Bubbling Pressure 42 = 0.09713178 0.09713178 0.09713178
Field Capacity 42 = 0.0395 0.0395 0.0395
Wilting Point 42 =  0.0202 0.0202 0.0202
Bulk Density 42 = 196.27 196.27 196.27
Vertical Conductivity 42 = 0.00001082 0.00001082 0.00001082
Thermal Conductivity 42 = 0.7548 0.7548 0.7548
Thermal Capacity 42 = 1180879 1180879 1180879
Mannings n 42 = 0.1000

Soil Description 43 = 2517833
Lateral Conductivity 43 = 0.00031000
Exponential Decrease 43 = 0.75623377
Depth Threshold 43 = 0.0500000000
Maximum Infiltration 43 = 0.00031000
Capillary Drive 43 = 0.28751640
Surface Albedo 43 = 0.25
Number of Soil Layers 43 = 3
Porosity 43 = 0.5278 0.5278 0.5278
Pore Size Distribution 43 = 0.57236275 0.57236275 0.57236275
Bubbling Pressure 43 = 0.08277912 0.08277912 0.08277912
Field Capacity 43 = 0.0968 0.0968 0.0968
Wilting Point 43 =  0.0374 0.0374 0.0374
Bulk Density 43 = 1251.40 1251.40 1251.40
Vertical Conductivity 43 = 0.00005755 0.00005755 0.00005755
Thermal Conductivity 43 = 0.7787 0.7787 0.7787
Thermal Capacity 43 = 1085194 1085194 1085194
Mannings n 43 = 0.0300

Soil Description 44 = 2517834
Lateral Conductivity 44 = 0.00029450
Exponential Decrease 44 = 0.97624682
Depth Threshold 44 = 0.0500000000
Maximum Infiltration 44 = 0.00029450
Capillary Drive 44 = 0.32148044
Surface Albedo 44 = 0.23
Number of Soil Layers 44 = 3
Porosity 44 = 0.6041 0.6041 0.6041
Pore Size Distribution 44 = 0.57561949 0.57561949 0.57561949
Bubbling Pressure 44 = 0.08065977 0.08065977 0.08065977
Field Capacity 44 = 0.0895 0.0895 0.0895
Wilting Point 44 =  0.0311 0.0311 0.0311
Bulk Density 44 = 1049.24 1049.24 1049.24
Vertical Conductivity 44 = 0.00005210 0.00005210 0.00005210
Thermal Conductivity 44 = 0.7822 0.7822 0.7822
Thermal Capacity 44 = 1071065 1071065 1071065
Mannings n 44 = 0.0300

Soil Description 45 =  612185
Lateral Conductivity 45 = 0.00001600
Exponential Decrease 45 = 0.00000000
Depth Threshold 45 = 0.0500000000
Maximum Infiltration 45 = 0.00001600
Capillary Drive 45 = 0.19202381
Surface Albedo 45 = 0.09
Number of Soil Layers 45 = 3
Porosity 45 = 0.7488 0.7488 0.7488
Pore Size Distribution 45 = 0.55120705 0.55120705 0.55120705
Bubbling Pressure 45 = 0.09654639 0.09654639 0.09654639
Field Capacity 45 = 0.0427 0.0427 0.0427
Wilting Point 45 =  0.0132 0.0132 0.0132
Bulk Density 45 = 665.60 665.60 665.60
Vertical Conductivity 45 = 0.00001600 0.00001600 0.00001600
Thermal Conductivity 45 = 0.7558 0.7558 0.7558
Thermal Capacity 45 = 1176976 1176976 1176976
Mannings n 45 = 0.0300

Soil Description 46 =  612189
Lateral Conductivity 46 = 0.00002800
Exponential Decrease 46 = 0.00000000
Depth Threshold 46 = 0.0500000000
Maximum Infiltration 46 = 0.00002800
Capillary Drive 46 = 0.32803947
Surface Albedo 46 = 0.32
Number of Soil Layers 46 = 3
Porosity 46 = 0.4927 0.4927 0.4927
Pore Size Distribution 46 = 0.56601984 0.56601984 0.56601984
Bubbling Pressure 46 = 0.08690683 0.08690683 0.08690683
Field Capacity 46 = 0.0751 0.0751 0.0751
Wilting Point 46 =  0.0227 0.0227 0.0227
Bulk Density 46 = 1344.40 1344.40 1344.40
Vertical Conductivity 46 = 0.00002800 0.00002800 0.00002800
Thermal Conductivity 46 = 0.7718 0.7718 0.7718
Thermal Capacity 46 = 1112712 1112712 1112712
Mannings n 46 = 0.0300

Soil Description 47 =  612193
Lateral Conductivity 47 = 0.00002100
Exponential Decrease 47 = 1.37657981
Depth Threshold 47 = 0.0500000000
Maximum Infiltration 47 = 0.00004340
Capillary Drive 47 = 0.26865789
Surface Albedo 47 = 0.25
Number of Soil Layers 47 = 3
Porosity 47 = 0.6116 0.6116 0.6116
Pore Size Distribution 47 = 0.54202034 0.54202034 0.54202034
Bubbling Pressure 47 = 0.10252472 0.10252472 0.10252472
Field Capacity 47 = 0.0900 0.0900 0.0900
Wilting Point 47 =  0.0253 0.0253 0.0253
Bulk Density 47 = 1029.33 1029.33 1029.33
Vertical Conductivity 47 = 0.00003972 0.00003972 0.00003972
Thermal Conductivity 47 = 0.7458 0.7458 0.7458
Thermal Capacity 47 = 1216831 1216831 1216831
Mannings n 47 = 0.0300

Soil Description 48 =  612194
Lateral Conductivity 48 = 0.00002800
Exponential Decrease 48 = 2.50419911
Depth Threshold 48 = 0.0500000000
Maximum Infiltration 48 = 0.00002800
Capillary Drive 48 = 0.26246053
Surface Albedo 48 = 0.34
Number of Soil Layers 48 = 3
Porosity 48 = 0.4337 0.4337 0.4337
Pore Size Distribution 48 = 0.54790937 0.54790937 0.54790937
Bubbling Pressure 48 = 0.09869238 0.09869238 0.09869238
Field Capacity 48 = 0.0703 0.0703 0.0703
Wilting Point 48 =  0.0237 0.0237 0.0237
Bulk Density 48 = 1500.76 1500.76 1500.76
Vertical Conductivity 48 = 0.00002006 0.00002006 0.00002006
Thermal Conductivity 48 = 0.7522 0.7522 0.7522
Thermal Capacity 48 = 1191283 1191283 1191283
Mannings n 48 = 0.0300

Soil Description 49 =  612195
Lateral Conductivity 49 = 0.00002800
Exponential Decrease 49 = 0.11613303
Depth Threshold 49 = 0.0500000000
Maximum Infiltration 49 = 0.00002800
Capillary Drive 49 = 0.32819737
Surface Albedo 49 = 0.26
Number of Soil Layers 49 = 3
Porosity 49 = 0.4967 0.4967 0.4967
Pore Size Distribution 49 = 0.56267554 0.56267554 0.56267554
Bubbling Pressure 49 = 0.08908316 0.08908316 0.08908316
Field Capacity 49 = 0.0736 0.0736 0.0736
Wilting Point 49 =  0.0208 0.0208 0.0208
Bulk Density 49 = 1333.67 1333.67 1333.67
Vertical Conductivity 49 = 0.00001443 0.00001443 0.00001443
Thermal Conductivity 49 = 0.7682 0.7682 0.7682
Thermal Capacity 49 = 1127221 1127221 1127221
Mannings n 49 = 0.0300

Soil Description 50 =  612197
Lateral Conductivity 50 = 0.00002800
Exponential Decrease 50 = 0.00000000
Depth Threshold 50 = 0.0500000000
Maximum Infiltration 50 = 0.00002800
Capillary Drive 50 = 0.36818421
Surface Albedo 50 = 0.30
Number of Soil Layers 50 = 3
Porosity 50 = 0.4992 0.4992 0.4992
Pore Size Distribution 50 = 0.54520455 0.54520455 0.54520455
Bubbling Pressure 50 = 0.10045257 0.10045257 0.10045257
Field Capacity 50 = 0.1091 0.1091 0.1091
Wilting Point 50 =  0.0317 0.0317 0.0317
Bulk Density 50 = 1327.25 1327.25 1327.25
Vertical Conductivity 50 = 0.00002800 0.00002800 0.00002800
Thermal Conductivity 50 = 0.7492 0.7492 0.7492
Thermal Capacity 50 = 1203017 1203017 1203017
Mannings n 50 = 0.0300

Soil Description 51 =  612198
Lateral Conductivity 51 = 0.00000685
Exponential Decrease 51 = 1.18066963
Depth Threshold 51 = 0.0500000000
Maximum Infiltration 51 = 0.00005680
Capillary Drive 51 = 0.41338312
Surface Albedo 51 = 0.02
Number of Soil Layers 51 = 3
Porosity 51 = 0.5060 0.5060 0.5060
Pore Size Distribution 51 = 0.53373921 0.53373921 0.53373921
Bubbling Pressure 51 = 0.10791375 0.10791375 0.10791375
Field Capacity 51 = 0.1421 0.1421 0.1421
Wilting Point 51 =  0.0388 0.0388 0.0388
Bulk Density 51 = 1309.11 1309.11 1309.11
Vertical Conductivity 51 = 0.00002058 0.00002058 0.00002058
Thermal Conductivity 51 = 0.7368 0.7368 0.7368
Thermal Capacity 51 = 1252758 1252758 1252758
Mannings n 51 = 0.0300

Soil Description 52 =  612199
Lateral Conductivity 52 = 0.00002800
Exponential Decrease 52 = 0.00000000
Depth Threshold 52 = 0.0500000000
Maximum Infiltration 52 = 0.00002800
Capillary Drive 52 = 0.30903349
Surface Albedo 52 = 0.20
Number of Soil Layers 52 = 3
Porosity 52 = 0.5167 0.5167 0.5167
Pore Size Distribution 52 = 0.54136393 0.54136393 0.54136393
Bubbling Pressure 52 = 0.10295189 0.10295189 0.10295189
Field Capacity 52 = 0.0805 0.0805 0.0805
Wilting Point 52 =  0.0289 0.0289 0.0289
Bulk Density 52 = 1280.69 1280.69 1280.69
Vertical Conductivity 52 = 0.00002800 0.00002800 0.00002800
Thermal Conductivity 52 = 0.7451 0.7451 0.7451
Thermal Capacity 52 = 1219679 1219679 1219679
Mannings n 52 = 0.0300

Soil Description 53 =  612201
Lateral Conductivity 53 = 0.00000615
Exponential Decrease 53 = 1.05938297
Depth Threshold 53 = 0.0500000000
Maximum Infiltration 53 = 0.00006320
Capillary Drive 53 = 0.41064286
Surface Albedo 53 = 0.02
Number of Soil Layers 53 = 3
Porosity 53 = 0.5055 0.5055 0.5055
Pore Size Distribution 53 = 0.53500377 0.53500377 0.53500377
Bubbling Pressure 53 = 0.10709082 0.10709082 0.10709082
Field Capacity 53 = 0.1457 0.1457 0.1457
Wilting Point 53 =  0.0403 0.0403 0.0403
Bulk Density 53 = 1310.50 1310.50 1310.50
Vertical Conductivity 53 = 0.00002095 0.00002095 0.00002095
Thermal Conductivity 53 = 0.7382 0.7382 0.7382
Thermal Capacity 53 = 1247272 1247272 1247272
Mannings n 53 = 0.0300

Soil Description 54 =  612204
Lateral Conductivity 54 = 0.00002800
Exponential Decrease 54 = 0.00000000
Depth Threshold 54 = 0.0500000000
Maximum Infiltration 54 = 0.00002800
Capillary Drive 54 = 0.42209704
Surface Albedo 54 = 0.24
Number of Soil Layers 54 = 3
Porosity 54 = 0.5675 0.5675 0.5675
Pore Size Distribution 54 = 0.56604794 0.56604794 0.56604794
Bubbling Pressure 54 = 0.08688854 0.08688854 0.08688854
Field Capacity 54 = 0.0875 0.0875 0.0875
Wilting Point 54 =  0.0248 0.0248 0.0248
Bulk Density 54 = 1146.09 1146.09 1146.09
Vertical Conductivity 54 = 0.00002800 0.00002800 0.00002800
Thermal Conductivity 54 = 0.7719 0.7719 0.7719
Thermal Capacity 54 = 1112590 1112590 1112590
Mannings n 54 = 0.0300

Soil Description 55 =  612205
Lateral Conductivity 55 = 0.00000900
Exponential Decrease 55 = 0.50729087
Depth Threshold 55 = 0.0500000000
Maximum Infiltration 55 = 0.00002800
Capillary Drive 55 = 0.35114145
Surface Albedo 55 = 0.15
Number of Soil Layers 55 = 3
Porosity 55 = 0.4427 0.4427 0.4427
Pore Size Distribution 55 = 0.53714111 0.53714111 0.53714111
Bubbling Pressure 55 = 0.10569993 0.10569993 0.10569993
Field Capacity 55 = 0.1496 0.1496 0.1496
Wilting Point 55 =  0.0530 0.0530 0.0530
Bulk Density 55 = 1476.78 1476.78 1476.78
Vertical Conductivity 55 = 0.00001819 0.00001819 0.00001819
Thermal Conductivity 55 = 0.7405 0.7405 0.7405
Thermal Capacity 55 = 1238000 1238000 1238000
Mannings n 55 = 0.0300

Soil Description 56 =  675750
Lateral Conductivity 56 = 0.00000459
Exponential Decrease 56 = 0.00000000
Depth Threshold 56 = 0.0500000000
Maximum Infiltration 56 = 0.00000459
Capillary Drive 56 = 0.16805921
Surface Albedo 56 = 0.00
Number of Soil Layers 56 = 3
Porosity 56 = 0.7431 0.7431 0.7431
Pore Size Distribution 56 = 0.48561539 0.48561539 0.48561539
Bubbling Pressure 56 = 0.13923077 0.13923077 0.13923077
Field Capacity 56 = 0.0668 0.0668 0.0668
Wilting Point 56 =  0.0217 0.0217 0.0217
Bulk Density 56 = 680.69 680.69 680.69
Vertical Conductivity 56 = 0.00000459 0.00000459 0.00000459
Thermal Conductivity 56 = 0.6846 0.6846 0.6846
Thermal Capacity 56 = 1461538 1461538 1461538
Mannings n 56 = 0.0300

Soil Description 57 =  675940
Lateral Conductivity 57 = 0.00001852
Exponential Decrease 57 = 0.82621941
Depth Threshold 57 = 0.0500000000
Maximum Infiltration 57 = 0.00003979
Capillary Drive 57 = 0.23618088
Surface Albedo 57 = 0.00
Number of Soil Layers 57 = 3
Porosity 57 = 0.5804 0.5804 0.5804
Pore Size Distribution 57 = 0.45412482 0.45412482 0.45412482
Bubbling Pressure 57 = 0.15972354 0.15972354 0.15972354
Field Capacity 57 = 0.1193 0.1193 0.1193
Wilting Point 57 =  0.0538 0.0538 0.0538
Bulk Density 57 = 1111.81 1111.81 1111.81
Vertical Conductivity 57 = 0.00002360 0.00002360 0.00002360
Thermal Conductivity 57 = 0.6505 0.6505 0.6505
Thermal Capacity 57 = 1598157 1598157 1598157
Mannings n 57 = 0.0300

Soil Description 58 =  675955
Lateral Conductivity 58 = 0.00000037
Exponential Decrease 58 = 3.27263914
Depth Threshold 58 = 0.0500000000
Maximum Infiltration 58 = 0.00000311
Capillary Drive 58 = 0.01146667
Surface Albedo 58 = 0.00
Number of Soil Layers 58 = 3
Porosity 58 = 0.9806 0.9806 0.9806
Pore Size Distribution 58 = 0.56884601 0.56884601 0.56884601
Bubbling Pressure 58 = 0.08506767 0.08506767 0.08506767
Field Capacity 58 = 0.0030 0.0030 0.0030
Wilting Point 58 =  0.0008 0.0008 0.0008
Bulk Density 58 = 51.50 51.50 51.50
Vertical Conductivity 58 = 0.00000243 0.00000243 0.00000243
Thermal Conductivity 58 = 0.7749 0.7749 0.7749
Thermal Capacity 58 = 1100451 1100451 1100451
Mannings n 58 = 0.1000

Soil Description 59 =  675972
Lateral Conductivity 59 = 0.00002243
Exponential Decrease 59 = 0.52918163
Depth Threshold 59 = 0.0500000000
Maximum Infiltration 59 = 0.00008440
Capillary Drive 59 = 0.33361118
Surface Albedo 59 = 0.00
Number of Soil Layers 59 = 3
Porosity 59 = 0.4977 0.4977 0.4977
Pore Size Distribution 59 = 0.56319285 0.56319285 0.56319285
Bubbling Pressure 59 = 0.08874652 0.08874652 0.08874652
Field Capacity 59 = 0.1005 0.1005 0.1005
Wilting Point 59 =  0.0385 0.0385 0.0385
Bulk Density 59 = 1330.99 1330.99 1330.99
Vertical Conductivity 59 = 0.00006047 0.00006047 0.00006047
Thermal Conductivity 59 = 0.7688 0.7688 0.7688
Thermal Capacity 59 = 1124977 1124977 1124977
Mannings n 59 = 0.0300

Soil Description 60 =  675973
Lateral Conductivity 60 = 0.00000615
Exponential Decrease 60 = 0.50000000
Depth Threshold 60 = 0.0500000000
Maximum Infiltration 60 = 0.00000957
Capillary Drive 60 = 0.13488698
Surface Albedo 60 = 0.24
Number of Soil Layers 60 = 3
Porosity 60 = 0.7192 0.7192 0.7192
Pore Size Distribution 60 = 0.50661316 0.50661316 0.50661316
Bubbling Pressure 60 = 0.12556628 0.12556628 0.12556628
Field Capacity 60 = 0.0520 0.0520 0.0520
Wilting Point 60 =  0.0204 0.0204 0.0204
Bulk Density 60 = 744.16 744.16 744.16
Vertical Conductivity 60 = 0.00000693 0.00000693 0.00000693
Thermal Conductivity 60 = 0.7074 0.7074 0.7074
Thermal Capacity 60 = 1370442 1370442 1370442
Mannings n 60 = 0.0300

Soil Description 61 =  675974 Loam
Lateral Conductivity 61 = 0.001700
Exponential Decrease 61 = 0.5
Depth Threshold 61 = 0.0500000000
Maximum Infiltration 61 = 0.017
Capillary Drive 61 = 0.08416652
Surface Albedo 61 = 0.24
Number of Soil Layers 61 = 3
Porosity 61 = 0.4 0.4 0.4
Pore Size Distribution 61 = 0.53180693 0.53180693 0.53180693
Bubbling Pressure 61 = 0.10917120 0.10917120 0.10917120
Field Capacity 61 = 0.0263 0.0263 0.0263
Wilting Point 61 =  0.0104 0.0104 0.0104
Bulk Density 61 = 591.59 591.59 591.59
Vertical Conductivity 61 = 0.0001764 0.0001764 0.0001764
Thermal Conductivity 61 = 0.7347 0.7347 0.7347
Thermal Capacity 61 = 1261141 1261141 1261141
Mannings n 61 = 0.0300

Soil Description 62 =  675975 sandy loam
Lateral Conductivity 62 = 0.00016340
Exponential Decrease 62 = 0.5
Depth Threshold 62 = 0.0500000000
Maximum Infiltration 62 = 0.00016340
Capillary Drive 62 = 0.34903431
Surface Albedo 62 = 0.24
Number of Soil Layers 62 = 3
Porosity 62 = 0.5 0.5 0.5
Pore Size Distribution 62 = 0.56482804 0.56482804 0.56482804
Bubbling Pressure 62 = 0.08768241 0.08768241 0.08768241
Field Capacity 62 = 0.1152 0.1152 0.1152
Wilting Point 62 =  0.0504 0.0504 0.0504
Bulk Density 62 = 756.04 756.04 756.04
Vertical Conductivity 62 = 0.00001 0.00001 0.00001
Thermal Conductivity 62 = 0.7705 0.7705 0.7705
Thermal Capacity 62 = 1117883 1117883 1117883
Mannings n 62 = 0.0300

Soil Description 63 =  675976
Lateral Conductivity 63 = 0.00001241
Exponential Decrease 63 = 2.65716458
Depth Threshold 63 = 0.0500000000
Maximum Infiltration 63 = 0.00001580
Capillary Drive 63 = 0.19660712
Surface Albedo 63 = 0.24
Number of Soil Layers 63 = 3
Porosity 63 = 0.7903 0.7903 0.7903
Pore Size Distribution 63 = 0.50214409 0.50214409 0.50214409
Bubbling Pressure 63 = 0.12847456 0.12847456 0.12847456
Field Capacity 63 = 0.0816 0.0816 0.0816
Wilting Point 63 =  0.0280 0.0280 0.0280
Bulk Density 63 = 555.61 555.61 555.61
Vertical Conductivity 63 = 0.00000855 0.00000855 0.00000855
Thermal Conductivity 63 = 0.7025 0.7025 0.7025
Thermal Capacity 63 = 1389830 1389830 1389830
Mannings n 63 = 0.0300

Soil Description 64 =  675980
Lateral Conductivity 64 = 0.00000735
Exponential Decrease 64 = 2.19334893
Depth Threshold 64 = 0.0500000000
Maximum Infiltration 64 = 0.00002767
Capillary Drive 64 = 0.02818244
Surface Albedo 64 = 0.24
Number of Soil Layers 64 = 3
Porosity 64 = 0.8253 0.8253 0.8253
Pore Size Distribution 64 = 0.54107650 0.54107650 0.54107650
Bubbling Pressure 64 = 0.10313894 0.10313894 0.10313894
Field Capacity 64 = 0.0108 0.0108 0.0108
Wilting Point 64 =  0.0040 0.0040 0.0040
Bulk Density 64 = 463.02 463.02 463.02
Vertical Conductivity 64 = 0.00002125 0.00002125 0.00002125
Thermal Conductivity 64 = 0.7448 0.7448 0.7448
Thermal Capacity 64 = 1220926 1220926 1220926
Mannings n 64 = 0.0300

Soil Description 65 =  676031 Bedrock
Lateral Conductivity 65 = 0.00000005
Exponential Decrease 65 = 2
Depth Threshold 65 = 0.0500000000
Maximum Infiltration 65 = 0.00000005
Capillary Drive 65 = 0.08017903
Surface Albedo 65 = 0.05
Number of Soil Layers 65 = 3
Porosity 65 = 0.01 0.01 0.01
Pore Size Distribution 65 = 0.55030749 0.55030749 0.55030749
Bubbling Pressure 65 = 0.09713178 0.09713178 0.09713178
Field Capacity 65 = 0.001 0.001 0.001
Wilting Point 65 =  0.0001 0.0001 0.0001
Bulk Density 65 = 196.27 196.27 196.27
Vertical Conductivity 65 = 0.00000005 0.00000005  0.00000005
Thermal Conductivity 65 = 0.7548 0.7548 0.7548
Thermal Capacity 65 = 1180879 1180879 1180879
Mannings n 65 = 0.1000

Soil Description 66 =  676071
Lateral Conductivity 66 = 0.00000005
Exponential Decrease 66 = 2.00000000
Depth Threshold 66 = 0.05000000
Maximum Infiltration 66 = 0.00000050
Capillary Drive 66 = 0.08000000
Surface Albedo 66 = 0.05
Number of Soil Layers 66 = 3
Porosity 66 = 0.0100 0.0100 0.0100
Pore Size Distribution 66 = 0.55000000 0.55000000 0.55000000
Bubbling Pressure 66 = 0.09700000 0.09700000 0.09700000
Field Capacity 66 = 0.0010 0.0010 0.0010
Wilting Point 66 =  0.0001 0.0001 0.0001
Bulk Density 66 = 196.00 196.00 196.00
Vertical Conductivity 66 = 0.00000005 0.00000005 0.00000005
Thermal Conductivity 66 = 0.7500 0.7500 0.7500
Thermal Capacity 66 = 1180879 1180879 1180879
Mannings n 66 = 0.1000

Soil Description 67 =   74726
Lateral Conductivity 67 = 0.00001789
Exponential Decrease 67 = 0.00000000
Depth Threshold 67 = 0.0500000000
Maximum Infiltration 67 = 0.00001789
Capillary Drive 67 = 0.19132310
Surface Albedo 67 = 0.18
Number of Soil Layers 67 = 3
Porosity 67 = 0.6691 0.6691 0.6691
Pore Size Distribution 67 = 0.52026812 0.52026812 0.52026812
Bubbling Pressure 67 = 0.11668018 0.11668018 0.11668018
Field Capacity 67 = 0.0994 0.0994 0.0994
Wilting Point 67 =  0.0306 0.0306 0.0306
Bulk Density 67 = 876.84 876.84 876.84
Vertical Conductivity 67 = 0.00001789 0.00001789 0.00001789
Thermal Conductivity 67 = 0.7222 0.7222 0.7222
Thermal Capacity 67 = 1311201 1311201 1311201
Mannings n 67 = 0.0300

Soil Description 68 =   74727
Lateral Conductivity 68 = 0.00000900
Exponential Decrease 68 = 0.77183572
Depth Threshold 68 = 0.0500000000
Maximum Infiltration 68 = 0.00000900
Capillary Drive 68 = 0.43723684
Surface Albedo 68 = 0.23
Number of Soil Layers 68 = 3
Porosity 68 = 0.3638 0.3638 0.3638
Pore Size Distribution 68 = 0.20746007 0.20746007 0.20746007
Bubbling Pressure 68 = 0.32024291 0.32024291 0.32024291
Field Capacity 68 = 0.3324 0.3324 0.3324
Wilting Point 68 =  0.2678 0.2678 0.2678
Bulk Density 68 = 1685.92 1685.92 1685.92
Vertical Conductivity 68 = 0.00000213 0.00000213 0.00000213
Thermal Conductivity 68 = 0.3829 0.3829 0.3829
Thermal Capacity 68 = 2668286 2668286 2668286
Mannings n 68 = 0.0300

Soil Description 69 =   74728
Lateral Conductivity 69 = 0.00000900
Exponential Decrease 69 = 0.40256106
Depth Threshold 69 = 0.0500000000
Maximum Infiltration 69 = 0.00009200
Capillary Drive 69 = 0.42763158
Surface Albedo 69 = 0.23
Number of Soil Layers 69 = 3
Porosity 69 = 0.5287 0.5287 0.5287
Pore Size Distribution 69 = 0.55116418 0.55116418 0.55116418
Bubbling Pressure 69 = 0.09657428 0.09657428 0.09657428
Field Capacity 69 = 0.1556 0.1556 0.1556
Wilting Point 69 =  0.0484 0.0484 0.0484
Bulk Density 69 = 1248.95 1248.95 1248.95
Vertical Conductivity 69 = 0.00004613 0.00004613 0.00004613
Thermal Conductivity 69 = 0.7557 0.7557 0.7557
Thermal Capacity 69 = 1177162 1177162 1177162
Mannings n 69 = 0.0300

Soil Description 70 =   74729
Lateral Conductivity 70 = 0.00000300
Exponential Decrease 70 = 0.82312685
Depth Threshold 70 = 0.0500000000
Maximum Infiltration 70 = 0.00000300
Capillary Drive 70 = 0.42467105
Surface Albedo 70 = 0.30
Number of Soil Layers 70 = 3
Porosity 70 = 0.3923 0.3923 0.3923
Pore Size Distribution 70 = 0.25064058 0.25064058 0.25064058
Bubbling Pressure 70 = 0.29214279 0.29214279 0.29214279
Field Capacity 70 = 0.3130 0.3130 0.3130
Wilting Point 70 =  0.1737 0.1737 0.1737
Bulk Density 70 = 1610.33 1610.33 1610.33
Vertical Conductivity 70 = 0.00000202 0.00000202 0.00000202
Thermal Conductivity 70 = 0.4298 0.4298 0.4298
Thermal Capacity 70 = 2480952 2480952 2480952
Mannings n 70 = 0.0300

Soil Description 71 =   74730
Lateral Conductivity 71 = 0.00000900
Exponential Decrease 71 = 0.65694313
Depth Threshold 71 = 0.0500000000
Maximum Infiltration 71 = 0.00002800
Capillary Drive 71 = 0.30144737
Surface Albedo 71 = 0.23
Number of Soil Layers 71 = 3
Porosity 71 = 0.4595 0.4595 0.4595
Pore Size Distribution 71 = 0.52573735 0.52573735 0.52573735
Bubbling Pressure 71 = 0.11312103 0.11312103 0.11312103
Field Capacity 71 = 0.1448 0.1448 0.1448
Wilting Point 71 =  0.0614 0.0614 0.0614
Bulk Density 71 = 1432.37 1432.37 1432.37
Vertical Conductivity 71 = 0.00002450 0.00002450 0.00002450
Thermal Conductivity 71 = 0.7281 0.7281 0.7281
Thermal Capacity 71 = 1287474 1287474 1287474
Mannings n 71 = 0.0300

Soil Description 72 =   74731
Lateral Conductivity 72 = 0.00000900
Exponential Decrease 72 = 0.00000000
Depth Threshold 72 = 0.0500000000
Maximum Infiltration 72 = 0.00000900
Capillary Drive 72 = 0.20842520
Surface Albedo 72 = 0.30
Number of Soil Layers 72 = 3
Porosity 72 = 0.6106 0.6106 0.6106
Pore Size Distribution 72 = 0.50257425 0.50257425 0.50257425
Bubbling Pressure 72 = 0.12819463 0.12819463 0.12819463
Field Capacity 72 = 0.1341 0.1341 0.1341
Wilting Point 72 =  0.0402 0.0402 0.0402
Bulk Density 72 = 1031.81 1031.81 1031.81
Vertical Conductivity 72 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 72 = 0.7030 0.7030 0.7030
Thermal Capacity 72 = 1387964 1387964 1387964
Mannings n 72 = 0.0300

Soil Description 73 =   74732
Lateral Conductivity 73 = 0.00009200
Exponential Decrease 73 = 1.33057199
Depth Threshold 73 = 0.0500000000
Maximum Infiltration 73 = 0.00030000
Capillary Drive 73 = 0.29138158
Surface Albedo 73 = 0.23
Number of Soil Layers 73 = 3
Porosity 73 = 0.3710 0.3710 0.3710
Pore Size Distribution 73 = 0.58026941 0.58026941 0.58026941
Bubbling Pressure 73 = 0.07763379 0.07763379 0.07763379
Field Capacity 73 = 0.0747 0.0747 0.0747
Wilting Point 73 =  0.0219 0.0219 0.0219
Bulk Density 73 = 1666.78 1666.78 1666.78
Vertical Conductivity 73 = 0.00015084 0.00015084 0.00015084
Thermal Conductivity 73 = 0.7873 0.7873 0.7873
Thermal Capacity 73 = 1050892 1050892 1050892
Mannings n 73 = 0.0300

Soil Description 74 =   74733
Lateral Conductivity 74 = 0.00002800
Exponential Decrease 74 = 0.38438308
Depth Threshold 74 = 0.0500000000
Maximum Infiltration 74 = 0.00030000
Capillary Drive 74 = 0.37934210
Surface Albedo 74 = 0.30
Number of Soil Layers 74 = 3
Porosity 74 = 0.3811 0.3811 0.3811
Pore Size Distribution 74 = 0.57646719 0.57646719 0.57646719
Bubbling Pressure 74 = 0.08010812 0.08010812 0.08010812
Field Capacity 74 = 0.1061 0.1061 0.1061
Wilting Point 74 =  0.0390 0.0390 0.0390
Bulk Density 74 = 1640.07 1640.07 1640.07
Vertical Conductivity 74 = 0.00015505 0.00015505 0.00015505
Thermal Conductivity 74 = 0.7832 0.7832 0.7832
Thermal Capacity 74 = 1067387 1067387 1067387
Mannings n 74 = 0.0300

Soil Description 75 =   74735
Lateral Conductivity 75 = 0.00000900
Exponential Decrease 75 = 0.00000000
Depth Threshold 75 = 0.0500000000
Maximum Infiltration 75 = 0.00000900
Capillary Drive 75 = 0.16945946
Surface Albedo 75 = 0.37
Number of Soil Layers 75 = 3
Porosity 75 = 0.6020 0.6020 0.6020
Pore Size Distribution 75 = 0.49149918 0.49149918 0.49149918
Bubbling Pressure 75 = 0.13540184 0.13540184 0.13540184
Field Capacity 75 = 0.1206 0.1206 0.1206
Wilting Point 75 =  0.0385 0.0385 0.0385
Bulk Density 75 = 1054.59 1054.59 1054.59
Vertical Conductivity 75 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 75 = 0.6910 0.6910 0.6910
Thermal Capacity 75 = 1436012 1436012 1436012
Mannings n 75 = 0.0300

Soil Description 76 =   74738
Lateral Conductivity 76 = 0.00000900
Exponential Decrease 76 = 0.77183572
Depth Threshold 76 = 0.0500000000
Maximum Infiltration 76 = 0.00000900
Capillary Drive 76 = 0.40756579
Surface Albedo 76 = 0.23
Number of Soil Layers 76 = 3
Porosity 76 = 0.3501 0.3501 0.3501
Pore Size Distribution 76 = 0.20746007 0.20746007 0.20746007
Bubbling Pressure 76 = 0.32024291 0.32024291 0.32024291
Field Capacity 76 = 0.3400 0.3400 0.3400
Wilting Point 76 =  0.3057 0.3057 0.3057
Bulk Density 76 = 1722.24 1722.24 1722.24
Vertical Conductivity 76 = 0.00000213 0.00000213 0.00000213
Thermal Conductivity 76 = 0.3829 0.3829 0.3829
Thermal Capacity 76 = 2668286 2668286 2668286
Mannings n 76 = 0.0300

Soil Description 77 =   74740
Lateral Conductivity 77 = 0.00000900
Exponential Decrease 77 = 0.18510245
Depth Threshold 77 = 0.0500000000
Maximum Infiltration 77 = 0.00000900
Capillary Drive 77 = 0.12309210
Surface Albedo 77 = 0.37
Number of Soil Layers 77 = 3
Porosity 77 = 0.3839 0.3839 0.3839
Pore Size Distribution 77 = 0.49942389 0.49942389 0.49942389
Bubbling Pressure 77 = 0.13024476 0.13024476 0.13024476
Field Capacity 77 = 0.1190 0.1190 0.1190
Wilting Point 77 =  0.0433 0.0433 0.0433
Bulk Density 77 = 1632.76 1632.76 1632.76
Vertical Conductivity 77 = 0.00000350 0.00000350 0.00000350
Thermal Conductivity 77 = 0.6996 0.6996 0.6996
Thermal Capacity 77 = 1401632 1401632 1401632
Mannings n 77 = 0.0300

Soil Description 78 =   74741
Lateral Conductivity 78 = 0.00000900
Exponential Decrease 78 = 0.18510245
Depth Threshold 78 = 0.0500000000
Maximum Infiltration 78 = 0.00000900
Capillary Drive 78 = 0.12309210
Surface Albedo 78 = 0.37
Number of Soil Layers 78 = 3
Porosity 78 = 0.3839 0.3839 0.3839
Pore Size Distribution 78 = 0.49942389 0.49942389 0.49942389
Bubbling Pressure 78 = 0.13024476 0.13024476 0.13024476
Field Capacity 78 = 0.1190 0.1190 0.1190
Wilting Point 78 =  0.0433 0.0433 0.0433
Bulk Density 78 = 1632.76 1632.76 1632.76
Vertical Conductivity 78 = 0.00000350 0.00000350 0.00000350
Thermal Conductivity 78 = 0.6996 0.6996 0.6996
Thermal Capacity 78 = 1401632 1401632 1401632
Mannings n 78 = 0.0300

Soil Description 79 =   74742
Lateral Conductivity 79 = 0.00009200
Exponential Decrease 79 = 0.51108578
Depth Threshold 79 = 0.0500000000
Maximum Infiltration 79 = 0.00009200
Capillary Drive 79 = 0.01250000
Surface Albedo 79 = 0.37
Number of Soil Layers 79 = 3
Porosity 79 = 0.5981 0.5981 0.5981
Pore Size Distribution 79 = 0.54230961 0.54230961 0.54230961
Bubbling Pressure 79 = 0.10233648 0.10233648 0.10233648
Field Capacity 79 = 0.0219 0.0219 0.0219
Wilting Point 79 =  0.0064 0.0064 0.0064
Bulk Density 79 = 1065.13 1065.13 1065.13
Vertical Conductivity 79 = 0.00001087 0.00001087 0.00001087
Thermal Conductivity 79 = 0.7461 0.7461 0.7461
Thermal Capacity 79 = 1215577 1215577 1215577
Mannings n 79 = 0.0300

Soil Description 80 =   74743
Lateral Conductivity 80 = 0.00000900
Exponential Decrease 80 = 2.81615117
Depth Threshold 80 = 0.0500000000
Maximum Infiltration 80 = 0.00000900
Capillary Drive 80 = 0.44657895
Surface Albedo 80 = 0.23
Number of Soil Layers 80 = 3
Porosity 80 = 0.4565 0.4565 0.4565
Pore Size Distribution 80 = 0.46900918 0.46900918 0.46900918
Bubbling Pressure 80 = 0.15003741 0.15003741 0.15003741
Field Capacity 80 = 0.2069 0.2069 0.2069
Wilting Point 80 =  0.0545 0.0545 0.0545
Bulk Density 80 = 1440.26 1440.26 1440.26
Vertical Conductivity 80 = 0.00000379 0.00000379 0.00000379
Thermal Conductivity 80 = 0.6666 0.6666 0.6666
Thermal Capacity 80 = 1533583 1533583 1533583
Mannings n 80 = 0.0300

Soil Description 81 =   74745
Lateral Conductivity 81 = 0.00000900
Exponential Decrease 81 = 0.65848578
Depth Threshold 81 = 0.0500000000
Maximum Infiltration 81 = 0.00009200
Capillary Drive 81 = 0.47848315
Surface Albedo 81 = 0.23
Number of Soil Layers 81 = 3
Porosity 81 = 0.5646 0.5646 0.5646
Pore Size Distribution 81 = 0.46986814 0.46986814 0.46986814
Bubbling Pressure 81 = 0.14947843 0.14947843 0.14947843
Field Capacity 81 = 0.1855 0.1855 0.1855
Wilting Point 81 =  0.0511 0.0511 0.0511
Bulk Density 81 = 1153.71 1153.71 1153.71
Vertical Conductivity 81 = 0.00002112 0.00002112 0.00002112
Thermal Conductivity 81 = 0.6675 0.6675 0.6675
Thermal Capacity 81 = 1529856 1529856 1529856
Mannings n 81 = 0.0300

Soil Description 82 =   74747
Lateral Conductivity 82 = 0.00000900
Exponential Decrease 82 = 0.00000000
Depth Threshold 82 = 0.0500000000
Maximum Infiltration 82 = 0.00000900
Capillary Drive 82 = 0.43822368
Surface Albedo 82 = 0.23
Number of Soil Layers 82 = 3
Porosity 82 = 0.4584 0.4584 0.4584
Pore Size Distribution 82 = 0.50922647 0.50922647 0.50922647
Bubbling Pressure 82 = 0.12386564 0.12386564 0.12386564
Field Capacity 82 = 0.2035 0.2035 0.2035
Wilting Point 82 =  0.0969 0.0969 0.0969
Bulk Density 82 = 1435.33 1435.33 1435.33
Vertical Conductivity 82 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 82 = 0.7102 0.7102 0.7102
Thermal Capacity 82 = 1359104 1359104 1359104
Mannings n 82 = 0.0300

Soil Description 83 =   74748
Lateral Conductivity 83 = 0.00000900
Exponential Decrease 83 = 0.17538222
Depth Threshold 83 = 0.0500000000
Maximum Infiltration 83 = 0.00000900
Capillary Drive 83 = 0.28190790
Surface Albedo 83 = 0.30
Number of Soil Layers 83 = 3
Porosity 83 = 0.4098 0.4098 0.4098
Pore Size Distribution 83 = 0.50787591 0.50787591 0.50787591
Bubbling Pressure 83 = 0.12474453 0.12474453 0.12474453
Field Capacity 83 = 0.1887 0.1887 0.1887
Wilting Point 83 =  0.0659 0.0659 0.0659
Bulk Density 83 = 1564.14 1564.14 1564.14
Vertical Conductivity 83 = 0.00000426 0.00000426 0.00000426
Thermal Conductivity 83 = 0.7088 0.7088 0.7088
Thermal Capacity 83 = 1364964 1364964 1364964
Mannings n 83 = 0.0300

Soil Description 84 =   74749
Lateral Conductivity 84 = 0.00002800
Exponential Decrease 84 = 0.58311384
Depth Threshold 84 = 0.0500000000
Maximum Infiltration 84 = 0.00009200
Capillary Drive 84 = 0.35342105
Surface Albedo 84 = 0.23
Number of Soil Layers 84 = 3
Porosity 84 = 0.4124 0.4124 0.4124
Pore Size Distribution 84 = 0.57925456 0.57925456 0.57925456
Bubbling Pressure 84 = 0.07829422 0.07829422 0.07829422
Field Capacity 84 = 0.0922 0.0922 0.0922
Wilting Point 84 =  0.0223 0.0223 0.0223
Bulk Density 84 = 1557.17 1557.17 1557.17
Vertical Conductivity 84 = 0.00008653 0.00008653 0.00008653
Thermal Conductivity 84 = 0.7862 0.7862 0.7862
Thermal Capacity 84 = 1055295 1055295 1055295
Mannings n 84 = 0.0300

Soil Description 85 =   74750
Lateral Conductivity 85 = 0.00000900
Exponential Decrease 85 = 0.17538222
Depth Threshold 85 = 0.0500000000
Maximum Infiltration 85 = 0.00000900
Capillary Drive 85 = 0.28190790
Surface Albedo 85 = 0.30
Number of Soil Layers 85 = 3
Porosity 85 = 0.4098 0.4098 0.4098
Pore Size Distribution 85 = 0.50787591 0.50787591 0.50787591
Bubbling Pressure 85 = 0.12474453 0.12474453 0.12474453
Field Capacity 85 = 0.1887 0.1887 0.1887
Wilting Point 85 =  0.0659 0.0659 0.0659
Bulk Density 85 = 1564.14 1564.14 1564.14
Vertical Conductivity 85 = 0.00000426 0.00000426 0.00000426
Thermal Conductivity 85 = 0.7088 0.7088 0.7088
Thermal Capacity 85 = 1364964 1364964 1364964
Mannings n 85 = 0.0300

Soil Description 86 =   74751
Lateral Conductivity 86 = 0.00000900
Exponential Decrease 86 = 0.17538222
Depth Threshold 86 = 0.0500000000
Maximum Infiltration 86 = 0.00000900
Capillary Drive 86 = 0.25151316
Surface Albedo 86 = 0.30
Number of Soil Layers 86 = 3
Porosity 86 = 0.4016 0.4016 0.4016
Pore Size Distribution 86 = 0.50787591 0.50787591 0.50787591
Bubbling Pressure 86 = 0.12474453 0.12474453 0.12474453
Field Capacity 86 = 0.1751 0.1751 0.1751
Wilting Point 86 =  0.0576 0.0576 0.0576
Bulk Density 86 = 1585.86 1585.86 1585.86
Vertical Conductivity 86 = 0.00000426 0.00000426 0.00000426
Thermal Conductivity 86 = 0.7088 0.7088 0.7088
Thermal Capacity 86 = 1364964 1364964 1364964
Mannings n 86 = 0.0300

Soil Description 87 =   74752
Lateral Conductivity 87 = 0.00000900
Exponential Decrease 87 = 0.17538222
Depth Threshold 87 = 0.0500000000
Maximum Infiltration 87 = 0.00000900
Capillary Drive 87 = 0.25151316
Surface Albedo 87 = 0.30
Number of Soil Layers 87 = 3
Porosity 87 = 0.4016 0.4016 0.4016
Pore Size Distribution 87 = 0.50787591 0.50787591 0.50787591
Bubbling Pressure 87 = 0.12474453 0.12474453 0.12474453
Field Capacity 87 = 0.1751 0.1751 0.1751
Wilting Point 87 =  0.0576 0.0576 0.0576
Bulk Density 87 = 1585.86 1585.86 1585.86
Vertical Conductivity 87 = 0.00000426 0.00000426 0.00000426
Thermal Conductivity 87 = 0.7088 0.7088 0.7088
Thermal Capacity 87 = 1364964 1364964 1364964
Mannings n 87 = 0.0300

Soil Description 88 =   74753
Lateral Conductivity 88 = 0.00000900
Exponential Decrease 88 = 0.00000000
Depth Threshold 88 = 0.0500000000
Maximum Infiltration 88 = 0.00000900
Capillary Drive 88 = 0.45625000
Surface Albedo 88 = 0.30
Number of Soil Layers 88 = 3
Porosity 88 = 0.4738 0.4738 0.4738
Pore Size Distribution 88 = 0.32774828 0.32774828 0.32774828
Bubbling Pressure 88 = 0.24196424 0.24196424 0.24196424
Field Capacity 88 = 0.2901 0.2901 0.2901
Wilting Point 88 =  0.1551 0.1551 0.1551
Bulk Density 88 = 1394.54 1394.54 1394.54
Vertical Conductivity 88 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 88 = 0.5134 0.5134 0.5134
Thermal Capacity 88 = 2146428 2146428 2146428
Mannings n 88 = 0.0300

Soil Description 89 =   74754
Lateral Conductivity 89 = 0.00000900
Exponential Decrease 89 = 0.00000000
Depth Threshold 89 = 0.0500000000
Maximum Infiltration 89 = 0.00000900
Capillary Drive 89 = 0.44243421
Surface Albedo 89 = 0.16
Number of Soil Layers 89 = 3
Porosity 89 = 0.4577 0.4577 0.4577
Pore Size Distribution 89 = 0.23226779 0.23226779 0.23226779
Bubbling Pressure 89 = 0.30409905 0.30409905 0.30409905
Field Capacity 89 = 0.3050 0.3050 0.3050
Wilting Point 89 =  0.2205 0.2205 0.2205
Bulk Density 89 = 1437.04 1437.04 1437.04
Vertical Conductivity 89 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 89 = 0.4098 0.4098 0.4098
Thermal Capacity 89 = 2560660 2560660 2560660
Mannings n 89 = 0.0300

Soil Description 90 =   74755
Lateral Conductivity 90 = 0.00000900
Exponential Decrease 90 = 0.00000000
Depth Threshold 90 = 0.0500000000
Maximum Infiltration 90 = 0.00000900
Capillary Drive 90 = 0.44243421
Surface Albedo 90 = 0.16
Number of Soil Layers 90 = 3
Porosity 90 = 0.4577 0.4577 0.4577
Pore Size Distribution 90 = 0.23226779 0.23226779 0.23226779
Bubbling Pressure 90 = 0.30409905 0.30409905 0.30409905
Field Capacity 90 = 0.3050 0.3050 0.3050
Wilting Point 90 =  0.2205 0.2205 0.2205
Bulk Density 90 = 1437.04 1437.04 1437.04
Vertical Conductivity 90 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 90 = 0.4098 0.4098 0.4098
Thermal Capacity 90 = 2560660 2560660 2560660
Mannings n 90 = 0.0300

Soil Description 91 =   74756
Lateral Conductivity 91 = 0.00000900
Exponential Decrease 91 = 0.78759448
Depth Threshold 91 = 0.0500000000
Maximum Infiltration 91 = 0.00002800
Capillary Drive 91 = 0.25802632
Surface Albedo 91 = 0.16
Number of Soil Layers 91 = 3
Porosity 91 = 0.5132 0.5132 0.5132
Pore Size Distribution 91 = 0.53053865 0.53053865 0.53053865
Bubbling Pressure 91 = 0.10999654 0.10999654 0.10999654
Field Capacity 91 = 0.1178 0.1178 0.1178
Wilting Point 91 =  0.0368 0.0368 0.0368
Bulk Density 91 = 1290.13 1290.13 1290.13
Vertical Conductivity 91 = 0.00002075 0.00002075 0.00002075
Thermal Conductivity 91 = 0.7333 0.7333 0.7333
Thermal Capacity 91 = 1266644 1266644 1266644
Mannings n 91 = 0.0300

Soil Description 92 =   74757
Lateral Conductivity 92 = 0.00000900
Exponential Decrease 92 = 0.78759448
Depth Threshold 92 = 0.0500000000
Maximum Infiltration 92 = 0.00002800
Capillary Drive 92 = 0.25802632
Surface Albedo 92 = 0.16
Number of Soil Layers 92 = 3
Porosity 92 = 0.5132 0.5132 0.5132
Pore Size Distribution 92 = 0.53053865 0.53053865 0.53053865
Bubbling Pressure 92 = 0.10999654 0.10999654 0.10999654
Field Capacity 92 = 0.1178 0.1178 0.1178
Wilting Point 92 =  0.0368 0.0368 0.0368
Bulk Density 92 = 1290.13 1290.13 1290.13
Vertical Conductivity 92 = 0.00002075 0.00002075 0.00002075
Thermal Conductivity 92 = 0.7333 0.7333 0.7333
Thermal Capacity 92 = 1266644 1266644 1266644
Mannings n 92 = 0.0300

Soil Description 93 =   74758
Lateral Conductivity 93 = 0.00002800
Exponential Decrease 93 = 0.31702665
Depth Threshold 93 = 0.0500000000
Maximum Infiltration 93 = 0.00030000
Capillary Drive 93 = 0.07190789
Surface Albedo 93 = 0.16
Number of Soil Layers 93 = 3
Porosity 93 = 0.4327 0.4327 0.4327
Pore Size Distribution 93 = 0.56589591 0.56589591 0.56589591
Bubbling Pressure 93 = 0.08698747 0.08698747 0.08698747
Field Capacity 93 = 0.0556 0.0556 0.0556
Wilting Point 93 =  0.0190 0.0190 0.0190
Bulk Density 93 = 1503.22 1503.22 1503.22
Vertical Conductivity 93 = 0.00022305 0.00022305 0.00022305
Thermal Conductivity 93 = 0.7717 0.7717 0.7717
Thermal Capacity 93 = 1113250 1113250 1113250
Mannings n 93 = 0.0300

Soil Description 94 =   74759
Lateral Conductivity 94 = 0.00002800
Exponential Decrease 94 = 0.32340688
Depth Threshold 94 = 0.0500000000
Maximum Infiltration 94 = 0.00030000
Capillary Drive 94 = 0.07894737
Surface Albedo 94 = 0.16
Number of Soil Layers 94 = 3
Porosity 94 = 0.4395 0.4395 0.4395
Pore Size Distribution 94 = 0.56357261 0.56357261 0.56357261
Bubbling Pressure 94 = 0.08849938 0.08849938 0.08849938
Field Capacity 94 = 0.0631 0.0631 0.0631
Wilting Point 94 =  0.0206 0.0206 0.0206
Bulk Density 94 = 1485.26 1485.26 1485.26
Vertical Conductivity 94 = 0.00021410 0.00021410 0.00021410
Thermal Conductivity 94 = 0.7692 0.7692 0.7692
Thermal Capacity 94 = 1123329 1123329 1123329
Mannings n 94 = 0.0300

Soil Description 95 =   74760
Lateral Conductivity 95 = 0.00002800
Exponential Decrease 95 = 0.58311384
Depth Threshold 95 = 0.0500000000
Maximum Infiltration 95 = 0.00009200
Capillary Drive 95 = 0.35342105
Surface Albedo 95 = 0.23
Number of Soil Layers 95 = 3
Porosity 95 = 0.4124 0.4124 0.4124
Pore Size Distribution 95 = 0.57925456 0.57925456 0.57925456
Bubbling Pressure 95 = 0.07829422 0.07829422 0.07829422
Field Capacity 95 = 0.0922 0.0922 0.0922
Wilting Point 95 =  0.0223 0.0223 0.0223
Bulk Density 95 = 1557.17 1557.17 1557.17
Vertical Conductivity 95 = 0.00008653 0.00008653 0.00008653
Thermal Conductivity 95 = 0.7862 0.7862 0.7862
Thermal Capacity 95 = 1055295 1055295 1055295
Mannings n 95 = 0.0300

Soil Description 96 =   74761
Lateral Conductivity 96 = 0.00000900
Exponential Decrease 96 = 0.73952125
Depth Threshold 96 = 0.0500000000
Maximum Infiltration 96 = 0.00000900
Capillary Drive 96 = 0.24782895
Surface Albedo 96 = 0.30
Number of Soil Layers 96 = 3
Porosity 96 = 0.7571 0.7571 0.7571
Pore Size Distribution 96 = 0.28190627 0.28190627 0.28190627
Bubbling Pressure 96 = 0.27179635 0.27179635 0.27179635
Field Capacity 96 = 0.2738 0.2738 0.2738
Wilting Point 96 =  0.1183 0.1183 0.1183
Bulk Density 96 = 643.75 643.75 643.75
Vertical Conductivity 96 = 0.00000730 0.00000730 0.00000730
Thermal Conductivity 96 = 0.4637 0.4637 0.4637
Thermal Capacity 96 = 2345309 2345309 2345309
Mannings n 96 = 0.0300

Soil Description 97 =   74762
Lateral Conductivity 97 = 0.00002800
Exponential Decrease 97 = 0.31066418
Depth Threshold 97 = 0.0500000000
Maximum Infiltration 97 = 0.00030000
Capillary Drive 97 = 0.11125000
Surface Albedo 97 = 0.30
Number of Soil Layers 97 = 3
Porosity 97 = 0.3905 0.3905 0.3905
Pore Size Distribution 97 = 0.57780960 0.57780960 0.57780960
Bubbling Pressure 97 = 0.07923454 0.07923454 0.07923454
Field Capacity 97 = 0.0488 0.0488 0.0488
Wilting Point 97 =  0.0129 0.0129 0.0129
Bulk Density 97 = 1615.20 1615.20 1615.20
Vertical Conductivity 97 = 0.00023274 0.00023274 0.00023274
Thermal Conductivity 97 = 0.7846 0.7846 0.7846
Thermal Capacity 97 = 1061564 1061564 1061564
Mannings n 97 = 0.0300

Soil Description 98 =   74763
Lateral Conductivity 98 = 0.00000900
Exponential Decrease 98 = 0.00000000
Depth Threshold 98 = 0.0500000000
Maximum Infiltration 98 = 0.00000900
Capillary Drive 98 = 0.15901316
Surface Albedo 98 = 0.30
Number of Soil Layers 98 = 3
Porosity 98 = 0.5843 0.5843 0.5843
Pore Size Distribution 98 = 0.50392490 0.50392490 0.50392490
Bubbling Pressure 98 = 0.12731569 0.12731569 0.12731569
Field Capacity 98 = 0.1048 0.1048 0.1048
Wilting Point 98 =  0.0294 0.0294 0.0294
Bulk Density 98 = 1101.51 1101.51 1101.51
Vertical Conductivity 98 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 98 = 0.7045 0.7045 0.7045
Thermal Capacity 98 = 1382105 1382105 1382105
Mannings n 98 = 0.0300

Soil Description 99 =   74764
Lateral Conductivity 99 = 0.00000900
Exponential Decrease 99 = 0.00000000
Depth Threshold 99 = 0.0500000000
Maximum Infiltration 99 = 0.00000900
Capillary Drive 99 = 0.15901316
Surface Albedo 99 = 0.30
Number of Soil Layers 99 = 3
Porosity 99 = 0.5843 0.5843 0.5843
Pore Size Distribution 99 = 0.50392490 0.50392490 0.50392490
Bubbling Pressure 99 = 0.12731569 0.12731569 0.12731569
Field Capacity 99 = 0.1048 0.1048 0.1048
Wilting Point 99 =  0.0294 0.0294 0.0294
Bulk Density 99 = 1101.51 1101.51 1101.51
Vertical Conductivity 99 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 99 = 0.7045 0.7045 0.7045
Thermal Capacity 99 = 1382105 1382105 1382105
Mannings n 99 = 0.0300

Soil Description 100 =   74765
Lateral Conductivity 100 = 0.00000900
Exponential Decrease 100 = 0.00000000
Depth Threshold 100 = 0.0500000000
Maximum Infiltration 100 = 0.00000900
Capillary Drive 100 = 0.16977528
Surface Albedo 100 = 0.37
Number of Soil Layers 100 = 3
Porosity 100 = 0.5767 0.5767 0.5767
Pore Size Distribution 100 = 0.50967857 0.50967857 0.50967857
Bubbling Pressure 100 = 0.12357143 0.12357143 0.12357143
Field Capacity 100 = 0.1203 0.1203 0.1203
Wilting Point 100 =  0.0385 0.0385 0.0385
Bulk Density 100 = 1121.80 1121.80 1121.80
Vertical Conductivity 100 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 100 = 0.7107 0.7107 0.7107
Thermal Capacity 100 = 1357143 1357143 1357143
Mannings n 100 = 0.0300

Soil Description 101 =   74766
Lateral Conductivity 101 = 0.00000900
Exponential Decrease 101 = 0.00000000
Depth Threshold 101 = 0.0500000000
Maximum Infiltration 101 = 0.00000900
Capillary Drive 101 = 0.18370370
Surface Albedo 101 = 0.30
Number of Soil Layers 101 = 3
Porosity 101 = 0.6231 0.6231 0.6231
Pore Size Distribution 101 = 0.49014889 0.49014889 0.49014889
Bubbling Pressure 101 = 0.13628055 0.13628055 0.13628055
Field Capacity 101 = 0.1194 0.1194 0.1194
Wilting Point 101 =  0.0330 0.0330 0.0330
Bulk Density 101 = 998.89 998.89 998.89
Vertical Conductivity 101 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 101 = 0.6895 0.6895 0.6895
Thermal Capacity 101 = 1441870 1441870 1441870
Mannings n 101 = 0.0300

Soil Description 102 =   74767
Lateral Conductivity 102 = 0.00000900
Exponential Decrease 102 = 0.32159724
Depth Threshold 102 = 0.0500000000
Maximum Infiltration 102 = 0.00009200
Capillary Drive 102 = 0.35756579
Surface Albedo 102 = 0.30
Number of Soil Layers 102 = 3
Porosity 102 = 0.4202 0.4202 0.4202
Pore Size Distribution 102 = 0.54590225 0.54590225 0.54590225
Bubbling Pressure 102 = 0.09999854 0.09999854 0.09999854
Field Capacity 102 = 0.1468 0.1468 0.1468
Wilting Point 102 =  0.0662 0.0662 0.0662
Bulk Density 102 = 1536.58 1536.58 1536.58
Vertical Conductivity 102 = 0.00006961 0.00006961 0.00006961
Thermal Conductivity 102 = 0.7500 0.7500 0.7500
Thermal Capacity 102 = 1199990 1199990 1199990
Mannings n 102 = 0.0300

Soil Description 103 =   74771
Lateral Conductivity 103 = 0.00000900
Exponential Decrease 103 = 0.00000000
Depth Threshold 103 = 0.0500000000
Maximum Infiltration 103 = 0.00000900
Capillary Drive 103 = 0.18185484
Surface Albedo 103 = 0.30
Number of Soil Layers 103 = 3
Porosity 103 = 0.5696 0.5696 0.5696
Pore Size Distribution 103 = 0.45598943 0.45598943 0.45598943
Bubbling Pressure 103 = 0.15851013 0.15851013 0.15851013
Field Capacity 103 = 0.1214 0.1214 0.1214
Wilting Point 103 =  0.0368 0.0368 0.0368
Bulk Density 103 = 1140.48 1140.48 1140.48
Vertical Conductivity 103 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 103 = 0.6525 0.6525 0.6525
Thermal Capacity 103 = 1590068 1590068 1590068
Mannings n 103 = 0.0300

Soil Description 104 =   74775
Lateral Conductivity 104 = 0.00000900
Exponential Decrease 104 = 0.52337250
Depth Threshold 104 = 0.0500000000
Maximum Infiltration 104 = 0.00000900
Capillary Drive 104 = 0.30118421
Surface Albedo 104 = 0.30
Number of Soil Layers 104 = 3
Porosity 104 = 0.6266 0.6266 0.6266
Pore Size Distribution 104 = 0.23823379 0.23823379 0.23823379
Bubbling Pressure 104 = 0.30021662 0.30021662 0.30021662
Field Capacity 104 = 0.3314 0.3314 0.3314
Wilting Point 104 =  0.2516 0.2516 0.2516
Bulk Density 104 = 989.61 989.61 989.61
Vertical Conductivity 104 = 0.00000320 0.00000320 0.00000320
Thermal Conductivity 104 = 0.4163 0.4163 0.4163
Thermal Capacity 104 = 2534777 2534777 2534777
Mannings n 104 = 0.0300

Soil Description 105 =   74776
Lateral Conductivity 105 = 0.00000900
Exponential Decrease 105 = 0.91476788
Depth Threshold 105 = 0.0500000000
Maximum Infiltration 105 = 0.00000900
Capillary Drive 105 = 0.44592105
Surface Albedo 105 = 0.37
Number of Soil Layers 105 = 3
Porosity 105 = 0.4558 0.4558 0.4558
Pore Size Distribution 105 = 0.28889602 0.28889602 0.28889602
Bubbling Pressure 105 = 0.26724771 0.26724771 0.26724771
Field Capacity 105 = 0.2799 0.2799 0.2799
Wilting Point 105 =  0.1268 0.1268 0.1268
Bulk Density 105 = 1442.24 1442.24 1442.24
Vertical Conductivity 105 = 0.00000187 0.00000187 0.00000187
Thermal Conductivity 105 = 0.4713 0.4713 0.4713
Thermal Capacity 105 = 2314985 2314985 2314985
Mannings n 105 = 0.0300

Soil Description 106 =   74777
Lateral Conductivity 106 = 0.00000300
Exponential Decrease 106 = 0.00000000
Depth Threshold 106 = 0.0500000000
Maximum Infiltration 106 = 0.00000300
Capillary Drive 106 = 0.67000000
Surface Albedo 106 = 0.23
Number of Soil Layers 106 = 3
Porosity 106 = 0.7057 0.7057 0.7057
Pore Size Distribution 106 = 0.38122008 0.38122008 0.38122008
Bubbling Pressure 106 = 0.20716697 0.20716697 0.20716697
Field Capacity 106 = 0.3285 0.3285 0.3285
Wilting Point 106 =  0.2059 0.2059 0.2059
Bulk Density 106 = 780.00 780.00 780.00
Vertical Conductivity 106 = 0.00000300 0.00000300 0.00000300
Thermal Conductivity 106 = 0.5714 0.5714 0.5714
Thermal Capacity 106 = 1914446 1914446 1914446
Mannings n 106 = 0.0300

Soil Description 107 =   74778
Lateral Conductivity 107 = 0.00036160
Exponential Decrease 107 = 0.17500818
Depth Threshold 107 = 0.0500000000
Maximum Infiltration 107 = 0.00036474
Capillary Drive 107 = 0.32771411
Surface Albedo 107 = 0.22
Number of Soil Layers 107 = 3
Porosity 107 = 0.5055 0.5055 0.5055
Pore Size Distribution 107 = 0.55024757 0.55024757 0.55024757
Bubbling Pressure 107 = 0.09717078 0.09717078 0.09717078
Field Capacity 107 = 0.0908 0.0908 0.0908
Wilting Point 107 =  0.0344 0.0344 0.0344
Bulk Density 107 = 1310.38 1310.38 1310.38
Vertical Conductivity 107 = 0.00004029 0.00004029 0.00004029
Thermal Conductivity 107 = 0.7547 0.7547 0.7547
Thermal Capacity 107 = 1181139 1181139 1181139
Mannings n 107 = 0.0300

Soil Description 108 =   74779
Lateral Conductivity 108 = 0.00036160
Exponential Decrease 108 = 0.19833682
Depth Threshold 108 = 0.0500000000
Maximum Infiltration 108 = 0.00036474
Capillary Drive 108 = 0.31643199
Surface Albedo 108 = 0.23
Number of Soil Layers 108 = 3
Porosity 108 = 0.5081 0.5081 0.5081
Pore Size Distribution 108 = 0.54273446 0.54273446 0.54273446
Bubbling Pressure 108 = 0.10206000 0.10206000 0.10206000
Field Capacity 108 = 0.0996 0.0996 0.0996
Wilting Point 108 =  0.0376 0.0376 0.0376
Bulk Density 108 = 1303.42 1303.42 1303.42
Vertical Conductivity 108 = 0.00004061 0.00004061 0.00004061
Thermal Conductivity 108 = 0.7466 0.7466 0.7466
Thermal Capacity 108 = 1213733 1213733 1213733
Mannings n 108 = 0.0300

Soil Description 109 =   74780
Lateral Conductivity 109 = 0.00036160
Exponential Decrease 109 = 0.21688178
Depth Threshold 109 = 0.0500000000
Maximum Infiltration 109 = 0.00036474
Capillary Drive 109 = 0.31566901
Surface Albedo 109 = 0.24
Number of Soil Layers 109 = 3
Porosity 109 = 0.5199 0.5199 0.5199
Pore Size Distribution 109 = 0.54215334 0.54215334 0.54215334
Bubbling Pressure 109 = 0.10243817 0.10243817 0.10243817
Field Capacity 109 = 0.1026 0.1026 0.1026
Wilting Point 109 =  0.0390 0.0390 0.0390
Bulk Density 109 = 1272.32 1272.32 1272.32
Vertical Conductivity 109 = 0.00004313 0.00004313 0.00004313
Thermal Conductivity 109 = 0.7459 0.7459 0.7459
Thermal Capacity 109 = 1216254 1216254 1216254
Mannings n 109 = 0.0300

Soil Description 110 =   74781
Lateral Conductivity 110 = 0.00035435
Exponential Decrease 110 = 0.21254035
Depth Threshold 110 = 0.0500000000
Maximum Infiltration 110 = 0.00035685
Capillary Drive 110 = 0.30200854
Surface Albedo 110 = 0.23
Number of Soil Layers 110 = 3
Porosity 110 = 0.5508 0.5508 0.5508
Pore Size Distribution 110 = 0.55092706 0.55092706 0.55092706
Bubbling Pressure 110 = 0.09672860 0.09672860 0.09672860
Field Capacity 110 = 0.0895 0.0895 0.0895
Wilting Point 110 =  0.0319 0.0319 0.0319
Bulk Density 110 = 1190.29 1190.29 1190.29
Vertical Conductivity 110 = 0.00004183 0.00004183 0.00004183
Thermal Conductivity 110 = 0.7555 0.7555 0.7555
Thermal Capacity 110 = 1178191 1178191 1178191
Mannings n 110 = 0.0300

Soil Description 111 =   74783
Lateral Conductivity 111 = 0.00000700
Exponential Decrease 111 = 0.00000000
Depth Threshold 111 = 0.0500000000
Maximum Infiltration 111 = 0.00000700
Capillary Drive 111 = 0.14337719
Surface Albedo 111 = 0.12
Number of Soil Layers 111 = 3
Porosity 111 = 0.5898 0.5898 0.5898
Pore Size Distribution 111 = 0.48667302 0.48667302 0.48667302
Bubbling Pressure 111 = 0.13854250 0.13854250 0.13854250
Field Capacity 111 = 0.1075 0.1075 0.1075
Wilting Point 111 =  0.0380 0.0380 0.0380
Bulk Density 111 = 1087.10 1087.10 1087.10
Vertical Conductivity 111 = 0.00000700 0.00000700 0.00000700
Thermal Conductivity 111 = 0.6858 0.6858 0.6858
Thermal Capacity 111 = 1456950 1456950 1456950
Mannings n 111 = 0.0300

Soil Description 112 =   74784
Lateral Conductivity 112 = 0.00001960
Exponential Decrease 112 = 1.49871468
Depth Threshold 112 = 0.0500000000
Maximum Infiltration 112 = 0.00006440
Capillary Drive 112 = 0.14746053
Surface Albedo 112 = 0.11
Number of Soil Layers 112 = 3
Porosity 112 = 0.6675 0.6675 0.6675
Pore Size Distribution 112 = 0.55048811 0.55048811 0.55048811
Bubbling Pressure 112 = 0.09701424 0.09701424 0.09701424
Field Capacity 112 = 0.0598 0.0598 0.0598
Wilting Point 112 =  0.0222 0.0222 0.0222
Bulk Density 112 = 881.08 881.08 881.08
Vertical Conductivity 112 = 0.00003669 0.00003669 0.00003669
Thermal Conductivity 112 = 0.7550 0.7550 0.7550
Thermal Capacity 112 = 1180095 1180095 1180095
Mannings n 112 = 0.0300

Soil Description 113 =   74785
Lateral Conductivity 113 = 0.00000500
Exponential Decrease 113 = 2.70210428
Depth Threshold 113 = 0.0500000000
Maximum Infiltration 113 = 0.00000922
Capillary Drive 113 = 0.29972222
Surface Albedo 113 = 0.17
Number of Soil Layers 113 = 3
Porosity 113 = 0.7558 0.7558 0.7558
Pore Size Distribution 113 = 0.54395396 0.54395396 0.54395396
Bubbling Pressure 113 = 0.10126640 0.10126640 0.10126640
Field Capacity 113 = 0.0956 0.0956 0.0956
Wilting Point 113 =  0.0331 0.0331 0.0331
Bulk Density 113 = 647.21 647.21 647.21
Vertical Conductivity 113 = 0.00000775 0.00000775 0.00000775
Thermal Conductivity 113 = 0.7479 0.7479 0.7479
Thermal Capacity 113 = 1208443 1208443 1208443
Mannings n 113 = 0.0300

Soil Description 114 =   74786
Lateral Conductivity 114 = 0.00000900
Exponential Decrease 114 = 0.16471214
Depth Threshold 114 = 0.0500000000
Maximum Infiltration 114 = 0.00000900
Capillary Drive 114 = 0.17269737
Surface Albedo 114 = 0.23
Number of Soil Layers 114 = 3
Porosity 114 = 0.4321 0.4321 0.4321
Pore Size Distribution 114 = 0.51620056 0.51620056 0.51620056
Bubbling Pressure 114 = 0.11932718 0.11932718 0.11932718
Field Capacity 114 = 0.1143 0.1143 0.1143
Wilting Point 114 =  0.0451 0.0451 0.0451
Bulk Density 114 = 1504.87 1504.87 1504.87
Vertical Conductivity 114 = 0.00000543 0.00000543 0.00000543
Thermal Conductivity 114 = 0.7178 0.7178 0.7178
Thermal Capacity 114 = 1328848 1328848 1328848
Mannings n 114 = 0.0300

Soil Description 115 =   74787
Lateral Conductivity 115 = 0.00000900
Exponential Decrease 115 = 0.16471214
Depth Threshold 115 = 0.0500000000
Maximum Infiltration 115 = 0.00000900
Capillary Drive 115 = 0.17269737
Surface Albedo 115 = 0.23
Number of Soil Layers 115 = 3
Porosity 115 = 0.4321 0.4321 0.4321
Pore Size Distribution 115 = 0.51620056 0.51620056 0.51620056
Bubbling Pressure 115 = 0.11932718 0.11932718 0.11932718
Field Capacity 115 = 0.1143 0.1143 0.1143
Wilting Point 115 =  0.0451 0.0451 0.0451
Bulk Density 115 = 1504.87 1504.87 1504.87
Vertical Conductivity 115 = 0.00000543 0.00000543 0.00000543
Thermal Conductivity 115 = 0.7178 0.7178 0.7178
Thermal Capacity 115 = 1328848 1328848 1328848
Mannings n 115 = 0.0300

Soil Description 116 =   74788
Lateral Conductivity 116 = 0.00000900
Exponential Decrease 116 = 0.16471214
Depth Threshold 116 = 0.0500000000
Maximum Infiltration 116 = 0.00000900
Capillary Drive 116 = 0.17269737
Surface Albedo 116 = 0.23
Number of Soil Layers 116 = 3
Porosity 116 = 0.4321 0.4321 0.4321
Pore Size Distribution 116 = 0.51620056 0.51620056 0.51620056
Bubbling Pressure 116 = 0.11932718 0.11932718 0.11932718
Field Capacity 116 = 0.1143 0.1143 0.1143
Wilting Point 116 =  0.0451 0.0451 0.0451
Bulk Density 116 = 1504.87 1504.87 1504.87
Vertical Conductivity 116 = 0.00000543 0.00000543 0.00000543
Thermal Conductivity 116 = 0.7178 0.7178 0.7178
Thermal Capacity 116 = 1328848 1328848 1328848
Mannings n 116 = 0.0300

Soil Description 117 =   74790
Lateral Conductivity 117 = 0.00000900
Exponential Decrease 117 = 0.21831965
Depth Threshold 117 = 0.0500000000
Maximum Infiltration 117 = 0.00030000
Capillary Drive 117 = 0.21789474
Surface Albedo 117 = 0.23
Number of Soil Layers 117 = 3
Porosity 117 = 0.4151 0.4151 0.4151
Pore Size Distribution 117 = 0.55881355 0.55881355 0.55881355
Bubbling Pressure 117 = 0.09159639 0.09159639 0.09159639
Field Capacity 117 = 0.1032 0.1032 0.1032
Wilting Point 117 =  0.0355 0.0355 0.0355
Bulk Density 117 = 1550.00 1550.00 1550.00
Vertical Conductivity 117 = 0.00018322 0.00018322 0.00018322
Thermal Conductivity 117 = 0.7640 0.7640 0.7640
Thermal Capacity 117 = 1143976 1143976 1143976
Mannings n 117 = 0.0300

Soil Description 118 =   74791
Lateral Conductivity 118 = 0.00000900
Exponential Decrease 118 = 0.21831965
Depth Threshold 118 = 0.0500000000
Maximum Infiltration 118 = 0.00030000
Capillary Drive 118 = 0.15052632
Surface Albedo 118 = 0.30
Number of Soil Layers 118 = 3
Porosity 118 = 0.3992 0.3992 0.3992
Pore Size Distribution 118 = 0.55939013 0.55939013 0.55939013
Bubbling Pressure 118 = 0.09122118 0.09122118 0.09122118
Field Capacity 118 = 0.0856 0.0856 0.0856
Wilting Point 118 =  0.0253 0.0253 0.0253
Bulk Density 118 = 1592.11 1592.11 1592.11
Vertical Conductivity 118 = 0.00018322 0.00018322 0.00018322
Thermal Conductivity 118 = 0.7646 0.7646 0.7646
Thermal Capacity 118 = 1141475 1141475 1141475
Mannings n 118 = 0.0300

Soil Description 119 =   74792
Lateral Conductivity 119 = 0.00000900
Exponential Decrease 119 = 0.19438238
Depth Threshold 119 = 0.0500000000
Maximum Infiltration 119 = 0.00030000
Capillary Drive 119 = 0.03434210
Surface Albedo 119 = 0.23
Number of Soil Layers 119 = 3
Porosity 119 = 0.3777 0.3777 0.3777
Pore Size Distribution 119 = 0.58242787 0.58242787 0.58242787
Bubbling Pressure 119 = 0.07622915 0.07622915 0.07622915
Field Capacity 119 = 0.0430 0.0430 0.0430
Wilting Point 119 =  0.0149 0.0149 0.0149
Bulk Density 119 = 1649.08 1649.08 1649.08
Vertical Conductivity 119 = 0.00026554 0.00026554 0.00026554
Thermal Conductivity 119 = 0.7896 0.7896 0.7896
Thermal Capacity 119 = 1041528 1041528 1041528
Mannings n 119 = 0.0300

Soil Description 120 =   74793
Lateral Conductivity 120 = 0.00000900
Exponential Decrease 120 = 0.46961713
Depth Threshold 120 = 0.0500000000
Maximum Infiltration 120 = 0.00000900
Capillary Drive 120 = 0.37690789
Surface Albedo 120 = 0.30
Number of Soil Layers 120 = 3
Porosity 120 = 0.4657 0.4657 0.4657
Pore Size Distribution 120 = 0.33303400 0.33303400 0.33303400
Bubbling Pressure 120 = 0.23852451 0.23852451 0.23852451
Field Capacity 120 = 0.2546 0.2546 0.2546
Wilting Point 120 =  0.1574 0.1574 0.1574
Bulk Density 120 = 1415.99 1415.99 1415.99
Vertical Conductivity 120 = 0.00000369 0.00000369 0.00000369
Thermal Conductivity 120 = 0.5191 0.5191 0.5191
Thermal Capacity 120 = 2123497 2123497 2123497
Mannings n 120 = 0.0300

Soil Description 121 =   74794
Lateral Conductivity 121 = 0.00000900
Exponential Decrease 121 = 0.16955742
Depth Threshold 121 = 0.0500000000
Maximum Infiltration 121 = 0.00000900
Capillary Drive 121 = 0.25263158
Surface Albedo 121 = 0.37
Number of Soil Layers 121 = 3
Porosity 121 = 0.4241 0.4241 0.4241
Pore Size Distribution 121 = 0.50027041 0.50027041 0.50027041
Bubbling Pressure 121 = 0.12969388 0.12969388 0.12969388
Field Capacity 121 = 0.1695 0.1695 0.1695
Wilting Point 121 =  0.0550 0.0550 0.0550
Bulk Density 121 = 1526.25 1526.25 1526.25
Vertical Conductivity 121 = 0.00000484 0.00000484 0.00000484
Thermal Conductivity 121 = 0.7005 0.7005 0.7005
Thermal Capacity 121 = 1397959 1397959 1397959
Mannings n 121 = 0.0300

Soil Description 122 =   74795
Lateral Conductivity 122 = 0.00000900
Exponential Decrease 122 = 0.16955742
Depth Threshold 122 = 0.0500000000
Maximum Infiltration 122 = 0.00000900
Capillary Drive 122 = 0.25263158
Surface Albedo 122 = 0.37
Number of Soil Layers 122 = 3
Porosity 122 = 0.4241 0.4241 0.4241
Pore Size Distribution 122 = 0.50027041 0.50027041 0.50027041
Bubbling Pressure 122 = 0.12969388 0.12969388 0.12969388
Field Capacity 122 = 0.1695 0.1695 0.1695
Wilting Point 122 =  0.0550 0.0550 0.0550
Bulk Density 122 = 1526.25 1526.25 1526.25
Vertical Conductivity 122 = 0.00000484 0.00000484 0.00000484
Thermal Conductivity 122 = 0.7005 0.7005 0.7005
Thermal Capacity 122 = 1397959 1397959 1397959
Mannings n 122 = 0.0300

Soil Description 123 =   74796
Lateral Conductivity 123 = 0.00000900
Exponential Decrease 123 = 1.25271573
Depth Threshold 123 = 0.0500000000
Maximum Infiltration 123 = 0.00000900
Capillary Drive 123 = 0.23040878
Surface Albedo 123 = 0.37
Number of Soil Layers 123 = 3
Porosity 123 = 0.4818 0.4818 0.4818
Pore Size Distribution 123 = 0.50005532 0.50005532 0.50005532
Bubbling Pressure 123 = 0.12983385 0.12983385 0.12983385
Field Capacity 123 = 0.1552 0.1552 0.1552
Wilting Point 123 =  0.0511 0.0511 0.0511
Bulk Density 123 = 1373.32 1373.32 1373.32
Vertical Conductivity 123 = 0.00000640 0.00000640 0.00000640
Thermal Conductivity 123 = 0.7003 0.7003 0.7003
Thermal Capacity 123 = 1398892 1398892 1398892
Mannings n 123 = 0.0300

Soil Description 124 =   74797
Lateral Conductivity 124 = 0.00000900
Exponential Decrease 124 = 0.83128818
Depth Threshold 124 = 0.0500000000
Maximum Infiltration 124 = 0.00000900
Capillary Drive 124 = 0.23082729
Surface Albedo 124 = 0.37
Number of Soil Layers 124 = 3
Porosity 124 = 0.4642 0.4642 0.4642
Pore Size Distribution 124 = 0.50289916 0.50289916 0.50289916
Bubbling Pressure 124 = 0.12798319 0.12798319 0.12798319
Field Capacity 124 = 0.1565 0.1565 0.1565
Wilting Point 124 =  0.0506 0.0506 0.0506
Bulk Density 124 = 1419.82 1419.82 1419.82
Vertical Conductivity 124 = 0.00000594 0.00000594 0.00000594
Thermal Conductivity 124 = 0.7034 0.7034 0.7034
Thermal Capacity 124 = 1386555 1386555 1386555
Mannings n 124 = 0.0300

Soil Description 125 =   74798
Lateral Conductivity 125 = 0.00000900
Exponential Decrease 125 = 1.25271573
Depth Threshold 125 = 0.0500000000
Maximum Infiltration 125 = 0.00000900
Capillary Drive 125 = 0.22156047
Surface Albedo 125 = 0.37
Number of Soil Layers 125 = 3
Porosity 125 = 0.4813 0.4813 0.4813
Pore Size Distribution 125 = 0.50398159 0.50398159 0.50398159
Bubbling Pressure 125 = 0.12727879 0.12727879 0.12727879
Field Capacity 125 = 0.1510 0.1510 0.1510
Wilting Point 125 =  0.0488 0.0488 0.0488
Bulk Density 125 = 1374.58 1374.58 1374.58
Vertical Conductivity 125 = 0.00000640 0.00000640 0.00000640
Thermal Conductivity 125 = 0.7045 0.7045 0.7045
Thermal Capacity 125 = 1381859 1381859 1381859
Mannings n 125 = 0.0300

Soil Description 126 =   74799
Lateral Conductivity 126 = 0.00002800
Exponential Decrease 126 = 0.59612849
Depth Threshold 126 = 0.0500000000
Maximum Infiltration 126 = 0.00009200
Capillary Drive 126 = 0.22368421
Surface Albedo 126 = 0.30
Number of Soil Layers 126 = 3
Porosity 126 = 0.3977 0.3977 0.3977
Pore Size Distribution 126 = 0.57136937 0.57136937 0.57136937
Bubbling Pressure 126 = 0.08342558 0.08342558 0.08342558
Field Capacity 126 = 0.0476 0.0476 0.0476
Wilting Point 126 =  0.0115 0.0115 0.0115
Bulk Density 126 = 1596.18 1596.18 1596.18
Vertical Conductivity 126 = 0.00008442 0.00008442 0.00008442
Thermal Conductivity 126 = 0.7776 0.7776 0.7776
Thermal Capacity 126 = 1089504 1089504 1089504
Mannings n 126 = 0.0300

Soil Description 127 =   74800
Lateral Conductivity 127 = 0.00000900
Exponential Decrease 127 = 0.46961713
Depth Threshold 127 = 0.0500000000
Maximum Infiltration 127 = 0.00000900
Capillary Drive 127 = 0.37690789
Surface Albedo 127 = 0.30
Number of Soil Layers 127 = 3
Porosity 127 = 0.4657 0.4657 0.4657
Pore Size Distribution 127 = 0.33303400 0.33303400 0.33303400
Bubbling Pressure 127 = 0.23852451 0.23852451 0.23852451
Field Capacity 127 = 0.2546 0.2546 0.2546
Wilting Point 127 =  0.1574 0.1574 0.1574
Bulk Density 127 = 1415.99 1415.99 1415.99
Vertical Conductivity 127 = 0.00000369 0.00000369 0.00000369
Thermal Conductivity 127 = 0.5191 0.5191 0.5191
Thermal Capacity 127 = 2123497 2123497 2123497
Mannings n 127 = 0.0300

Soil Description 128 =   74803
Lateral Conductivity 128 = 0.00000900
Exponential Decrease 128 = 0.00000000
Depth Threshold 128 = 0.0500000000
Maximum Infiltration 128 = 0.00000900
Capillary Drive 128 = 0.23171053
Surface Albedo 128 = 0.09
Number of Soil Layers 128 = 3
Porosity 128 = 0.5219 0.5219 0.5219
Pore Size Distribution 128 = 0.50470504 0.50470504 0.50470504
Bubbling Pressure 128 = 0.12680800 0.12680800 0.12680800
Field Capacity 128 = 0.1423 0.1423 0.1423
Wilting Point 128 =  0.0417 0.0417 0.0417
Bulk Density 128 = 1266.97 1266.97 1266.97
Vertical Conductivity 128 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 128 = 0.7053 0.7053 0.7053
Thermal Capacity 128 = 1378720 1378720 1378720
Mannings n 128 = 0.0300

Soil Description 129 =   74804
Lateral Conductivity 129 = 0.00000900
Exponential Decrease 129 = 0.46961713
Depth Threshold 129 = 0.0500000000
Maximum Infiltration 129 = 0.00000900
Capillary Drive 129 = 0.37690789
Surface Albedo 129 = 0.30
Number of Soil Layers 129 = 3
Porosity 129 = 0.4657 0.4657 0.4657
Pore Size Distribution 129 = 0.33303400 0.33303400 0.33303400
Bubbling Pressure 129 = 0.23852451 0.23852451 0.23852451
Field Capacity 129 = 0.2546 0.2546 0.2546
Wilting Point 129 =  0.1574 0.1574 0.1574
Bulk Density 129 = 1415.99 1415.99 1415.99
Vertical Conductivity 129 = 0.00000369 0.00000369 0.00000369
Thermal Conductivity 129 = 0.5191 0.5191 0.5191
Thermal Capacity 129 = 2123497 2123497 2123497
Mannings n 129 = 0.0300

Soil Description 130 =   74805
Lateral Conductivity 130 = 0.00000900
Exponential Decrease 130 = 0.00000000
Depth Threshold 130 = 0.0500000000
Maximum Infiltration 130 = 0.00000900
Capillary Drive 130 = 0.44578947
Surface Albedo 130 = 0.30
Number of Soil Layers 130 = 3
Porosity 130 = 0.4635 0.4635 0.4635
Pore Size Distribution 130 = 0.44388256 0.44388256 0.44388256
Bubbling Pressure 130 = 0.16638879 0.16638879 0.16638879
Field Capacity 130 = 0.2594 0.2594 0.2594
Wilting Point 130 =  0.1244 0.1244 0.1244
Bulk Density 130 = 1421.84 1421.84 1421.84
Vertical Conductivity 130 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 130 = 0.6394 0.6394 0.6394
Thermal Capacity 130 = 1642592 1642592 1642592
Mannings n 130 = 0.0300

Soil Description 131 =   74806
Lateral Conductivity 131 = 0.00000900
Exponential Decrease 131 = 0.00000000
Depth Threshold 131 = 0.0500000000
Maximum Infiltration 131 = 0.00000900
Capillary Drive 131 = 0.47736842
Surface Albedo 131 = 0.23
Number of Soil Layers 131 = 3
Porosity 131 = 0.6205 0.6205 0.6205
Pore Size Distribution 131 = 0.51856466 0.51856466 0.51856466
Bubbling Pressure 131 = 0.11778872 0.11778872 0.11778872
Field Capacity 131 = 0.1671 0.1671 0.1671
Wilting Point 131 =  0.0456 0.0456 0.0456
Bulk Density 131 = 1005.79 1005.79 1005.79
Vertical Conductivity 131 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 131 = 0.7204 0.7204 0.7204
Thermal Capacity 131 = 1318591 1318591 1318591
Mannings n 131 = 0.0300

Soil Description 132 =   74807
Lateral Conductivity 132 = 0.00000900
Exponential Decrease 132 = 0.00000000
Depth Threshold 132 = 0.0500000000
Maximum Infiltration 132 = 0.00000900
Capillary Drive 132 = 0.47736842
Surface Albedo 132 = 0.23
Number of Soil Layers 132 = 3
Porosity 132 = 0.6205 0.6205 0.6205
Pore Size Distribution 132 = 0.51856466 0.51856466 0.51856466
Bubbling Pressure 132 = 0.11778872 0.11778872 0.11778872
Field Capacity 132 = 0.1671 0.1671 0.1671
Wilting Point 132 =  0.0456 0.0456 0.0456
Bulk Density 132 = 1005.79 1005.79 1005.79
Vertical Conductivity 132 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 132 = 0.7204 0.7204 0.7204
Thermal Capacity 132 = 1318591 1318591 1318591
Mannings n 132 = 0.0300

Soil Description 133 =   74808
Lateral Conductivity 133 = 0.00000900
Exponential Decrease 133 = 0.00000000
Depth Threshold 133 = 0.0500000000
Maximum Infiltration 133 = 0.00000900
Capillary Drive 133 = 0.47085526
Surface Albedo 133 = 0.23
Number of Soil Layers 133 = 3
Porosity 133 = 0.6164 0.6164 0.6164
Pore Size Distribution 133 = 0.50967857 0.50967857 0.50967857
Bubbling Pressure 133 = 0.12357143 0.12357143 0.12357143
Field Capacity 133 = 0.1801 0.1801 0.1801
Wilting Point 133 =  0.0445 0.0445 0.0445
Bulk Density 133 = 1016.45 1016.45 1016.45
Vertical Conductivity 133 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 133 = 0.7107 0.7107 0.7107
Thermal Capacity 133 = 1357143 1357143 1357143
Mannings n 133 = 0.0300

Soil Description 134 =   74811
Lateral Conductivity 134 = 0.00033572
Exponential Decrease 134 = 0.43449774
Depth Threshold 134 = 0.0500000000
Maximum Infiltration 134 = 0.00033572
Capillary Drive 134 = 0.30633183
Surface Albedo 134 = 0.33
Number of Soil Layers 134 = 3
Porosity 134 = 0.5931 0.5931 0.5931
Pore Size Distribution 134 = 0.49602792 0.49602792 0.49602792
Bubbling Pressure 134 = 0.13245472 0.13245472 0.13245472
Field Capacity 134 = 0.1511 0.1511 0.1511
Wilting Point 134 =  0.0554 0.0554 0.0554
Bulk Density 134 = 1078.29 1078.29 1078.29
Vertical Conductivity 134 = 0.00004836 0.00004836 0.00004836
Thermal Conductivity 134 = 0.6959 0.6959 0.6959
Thermal Capacity 134 = 1416365 1416365 1416365
Mannings n 134 = 0.0300

Soil Description 135 =   74812
Lateral Conductivity 135 = 0.00033572
Exponential Decrease 135 = 0.43449774
Depth Threshold 135 = 0.0500000000
Maximum Infiltration 135 = 0.00033572
Capillary Drive 135 = 0.30633183
Surface Albedo 135 = 0.33
Number of Soil Layers 135 = 3
Porosity 135 = 0.5931 0.5931 0.5931
Pore Size Distribution 135 = 0.49602792 0.49602792 0.49602792
Bubbling Pressure 135 = 0.13245472 0.13245472 0.13245472
Field Capacity 135 = 0.1511 0.1511 0.1511
Wilting Point 135 =  0.0554 0.0554 0.0554
Bulk Density 135 = 1078.29 1078.29 1078.29
Vertical Conductivity 135 = 0.00004836 0.00004836 0.00004836
Thermal Conductivity 135 = 0.6959 0.6959 0.6959
Thermal Capacity 135 = 1416365 1416365 1416365
Mannings n 135 = 0.0300

Soil Description 136 =   74814
Lateral Conductivity 136 = 0.00000750
Exponential Decrease 136 = 0.00000000
Depth Threshold 136 = 0.0500000000
Maximum Infiltration 136 = 0.00000750
Capillary Drive 136 = 0.19144737
Surface Albedo 136 = 0.02
Number of Soil Layers 136 = 3
Porosity 136 = 0.6039 0.6039 0.6039
Pore Size Distribution 136 = 0.50274701 0.50274701 0.50274701
Bubbling Pressure 136 = 0.12808221 0.12808221 0.12808221
Field Capacity 136 = 0.1180 0.1180 0.1180
Wilting Point 136 =  0.0348 0.0348 0.0348
Bulk Density 136 = 1049.78 1049.78 1049.78
Vertical Conductivity 136 = 0.00000750 0.00000750 0.00000750
Thermal Conductivity 136 = 0.7032 0.7032 0.7032
Thermal Capacity 136 = 1387215 1387215 1387215
Mannings n 136 = 0.0300

Soil Description 137 =   74817
Lateral Conductivity 137 = 0.00000900
Exponential Decrease 137 = 0.20436979
Depth Threshold 137 = 0.0500000000
Maximum Infiltration 137 = 0.00000900
Capillary Drive 137 = 0.07407895
Surface Albedo 137 = 0.30
Number of Soil Layers 137 = 3
Porosity 137 = 0.3342 0.3342 0.3342
Pore Size Distribution 137 = 0.49763368 0.49763368 0.49763368
Bubbling Pressure 137 = 0.13140975 0.13140975 0.13140975
Field Capacity 137 = 0.1311 0.1311 0.1311
Wilting Point 137 =  0.0528 0.0528 0.0528
Bulk Density 137 = 1764.41 1764.41 1764.41
Vertical Conductivity 137 = 0.00000250 0.00000250 0.00000250
Thermal Conductivity 137 = 0.6977 0.6977 0.6977
Thermal Capacity 137 = 1409398 1409398 1409398
Mannings n 137 = 0.0300

Soil Description 138 =   74818
Lateral Conductivity 138 = 0.00000900
Exponential Decrease 138 = 0.20436979
Depth Threshold 138 = 0.0500000000
Maximum Infiltration 138 = 0.00000900
Capillary Drive 138 = 0.07407895
Surface Albedo 138 = 0.30
Number of Soil Layers 138 = 3
Porosity 138 = 0.3342 0.3342 0.3342
Pore Size Distribution 138 = 0.49763368 0.49763368 0.49763368
Bubbling Pressure 138 = 0.13140975 0.13140975 0.13140975
Field Capacity 138 = 0.1311 0.1311 0.1311
Wilting Point 138 =  0.0528 0.0528 0.0528
Bulk Density 138 = 1764.41 1764.41 1764.41
Vertical Conductivity 138 = 0.00000250 0.00000250 0.00000250
Thermal Conductivity 138 = 0.6977 0.6977 0.6977
Thermal Capacity 138 = 1409398 1409398 1409398
Mannings n 138 = 0.0300

Soil Description 139 =   74819
Lateral Conductivity 139 = 0.00000900
Exponential Decrease 139 = 0.22396495
Depth Threshold 139 = 0.0500000000
Maximum Infiltration 139 = 0.00030000
Capillary Drive 139 = 0.37328947
Surface Albedo 139 = 0.30
Number of Soil Layers 139 = 3
Porosity 139 = 0.4331 0.4331 0.4331
Pore Size Distribution 139 = 0.55408123 0.55408123 0.55408123
Bubbling Pressure 139 = 0.09467599 0.09467599 0.09467599
Field Capacity 139 = 0.1475 0.1475 0.1475
Wilting Point 139 =  0.0464 0.0464 0.0464
Bulk Density 139 = 1502.37 1502.37 1502.37
Vertical Conductivity 139 = 0.00016982 0.00016982 0.00016982
Thermal Conductivity 139 = 0.7589 0.7589 0.7589
Thermal Capacity 139 = 1164507 1164507 1164507
Mannings n 139 = 0.0300

Soil Description 140 =   74824
Lateral Conductivity 140 = 0.00000675
Exponential Decrease 140 = 0.00000000
Depth Threshold 140 = 0.0500000000
Maximum Infiltration 140 = 0.00000675
Capillary Drive 140 = 0.13703704
Surface Albedo 140 = 0.23
Number of Soil Layers 140 = 3
Porosity 140 = 0.6754 0.6754 0.6754
Pore Size Distribution 140 = 0.49421412 0.49421412 0.49421412
Bubbling Pressure 140 = 0.13363506 0.13363506 0.13363506
Field Capacity 140 = 0.0953 0.0953 0.0953
Wilting Point 140 =  0.0311 0.0311 0.0311
Bulk Density 140 = 860.28 860.28 860.28
Vertical Conductivity 140 = 0.00000675 0.00000675 0.00000675
Thermal Conductivity 140 = 0.6939 0.6939 0.6939
Thermal Capacity 140 = 1424234 1424234 1424234
Mannings n 140 = 0.0300

Soil Description 141 =   74825
Lateral Conductivity 141 = 0.00000650
Exponential Decrease 141 = 1.11485610
Depth Threshold 141 = 0.0500000000
Maximum Infiltration 141 = 0.00002022
Capillary Drive 141 = 0.13033260
Surface Albedo 141 = 0.07
Number of Soil Layers 141 = 3
Porosity 141 = 0.6466 0.6466 0.6466
Pore Size Distribution 141 = 0.52217457 0.52217457 0.52217457
Bubbling Pressure 141 = 0.11543954 0.11543954 0.11543954
Field Capacity 141 = 0.0746 0.0746 0.0746
Wilting Point 141 =  0.0255 0.0255 0.0255
Bulk Density 141 = 936.56 936.56 936.56
Vertical Conductivity 141 = 0.00001472 0.00001472 0.00001472
Thermal Conductivity 141 = 0.7243 0.7243 0.7243
Thermal Capacity 141 = 1302930 1302930 1302930
Mannings n 141 = 0.0300

Soil Description 142 =   74826
Lateral Conductivity 142 = 0.00000650
Exponential Decrease 142 = 0.00000000
Depth Threshold 142 = 0.0500000000
Maximum Infiltration 142 = 0.00000650
Capillary Drive 142 = 0.13196159
Surface Albedo 142 = 0.22
Number of Soil Layers 142 = 3
Porosity 142 = 0.6874 0.6874 0.6874
Pore Size Distribution 142 = 0.49421412 0.49421412 0.49421412
Bubbling Pressure 142 = 0.13363506 0.13363506 0.13363506
Field Capacity 142 = 0.0918 0.0918 0.0918
Wilting Point 142 =  0.0300 0.0300 0.0300
Bulk Density 142 = 828.42 828.42 828.42
Vertical Conductivity 142 = 0.00000650 0.00000650 0.00000650
Thermal Conductivity 142 = 0.6939 0.6939 0.6939
Thermal Capacity 142 = 1424234 1424234 1424234
Mannings n 142 = 0.0300

Soil Description 143 =   74827
Lateral Conductivity 143 = 0.00000900
Exponential Decrease 143 = 0.16703061
Depth Threshold 143 = 0.0500000000
Maximum Infiltration 143 = 0.00000900
Capillary Drive 143 = 0.11157895
Surface Albedo 143 = 0.30
Number of Soil Layers 143 = 3
Porosity 143 = 0.4236 0.4236 0.4236
Pore Size Distribution 143 = 0.52572111 0.52572111 0.52572111
Bubbling Pressure 143 = 0.11313160 0.11313160 0.11313160
Field Capacity 143 = 0.0915 0.0915 0.0915
Wilting Point 143 =  0.0403 0.0403 0.0403
Bulk Density 143 = 1527.37 1527.37 1527.37
Vertical Conductivity 143 = 0.00000514 0.00000514 0.00000514
Thermal Conductivity 143 = 0.7281 0.7281 0.7281
Thermal Capacity 143 = 1287544 1287544 1287544
Mannings n 143 = 0.0300

Soil Description 144 =   74828
Lateral Conductivity 144 = 0.00000900
Exponential Decrease 144 = 0.16703061
Depth Threshold 144 = 0.0500000000
Maximum Infiltration 144 = 0.00000900
Capillary Drive 144 = 0.11157895
Surface Albedo 144 = 0.30
Number of Soil Layers 144 = 3
Porosity 144 = 0.4236 0.4236 0.4236
Pore Size Distribution 144 = 0.52572111 0.52572111 0.52572111
Bubbling Pressure 144 = 0.11313160 0.11313160 0.11313160
Field Capacity 144 = 0.0915 0.0915 0.0915
Wilting Point 144 =  0.0403 0.0403 0.0403
Bulk Density 144 = 1527.37 1527.37 1527.37
Vertical Conductivity 144 = 0.00000514 0.00000514 0.00000514
Thermal Conductivity 144 = 0.7281 0.7281 0.7281
Thermal Capacity 144 = 1287544 1287544 1287544
Mannings n 144 = 0.0300

Soil Description 145 =   74829
Lateral Conductivity 145 = 0.00000900
Exponential Decrease 145 = 0.18005276
Depth Threshold 145 = 0.0500000000
Maximum Infiltration 145 = 0.00000900
Capillary Drive 145 = 0.25151316
Surface Albedo 145 = 0.44
Number of Soil Layers 145 = 3
Porosity 145 = 0.4194 0.4194 0.4194
Pore Size Distribution 145 = 0.49942389 0.49942389 0.49942389
Bubbling Pressure 145 = 0.13024476 0.13024476 0.13024476
Field Capacity 145 = 0.1807 0.1807 0.1807
Wilting Point 145 =  0.0661 0.0661 0.0661
Bulk Density 145 = 1538.55 1538.55 1538.55
Vertical Conductivity 145 = 0.00000386 0.00000386 0.00000386
Thermal Conductivity 145 = 0.6996 0.6996 0.6996
Thermal Capacity 145 = 1401632 1401632 1401632
Mannings n 145 = 0.0300

Soil Description 146 =   74830
Lateral Conductivity 146 = 0.00000900
Exponential Decrease 146 = 0.18005276
Depth Threshold 146 = 0.0500000000
Maximum Infiltration 146 = 0.00000900
Capillary Drive 146 = 0.25151316
Surface Albedo 146 = 0.44
Number of Soil Layers 146 = 3
Porosity 146 = 0.4194 0.4194 0.4194
Pore Size Distribution 146 = 0.49942389 0.49942389 0.49942389
Bubbling Pressure 146 = 0.13024476 0.13024476 0.13024476
Field Capacity 146 = 0.1807 0.1807 0.1807
Wilting Point 146 =  0.0661 0.0661 0.0661
Bulk Density 146 = 1538.55 1538.55 1538.55
Vertical Conductivity 146 = 0.00000386 0.00000386 0.00000386
Thermal Conductivity 146 = 0.6996 0.6996 0.6996
Thermal Capacity 146 = 1401632 1401632 1401632
Mannings n 146 = 0.0300

Soil Description 147 =   74831
Lateral Conductivity 147 = 0.00000900
Exponential Decrease 147 = 0.83517411
Depth Threshold 147 = 0.0500000000
Maximum Infiltration 147 = 0.00000900
Capillary Drive 147 = 0.23622491
Surface Albedo 147 = 0.41
Number of Soil Layers 147 = 3
Porosity 147 = 0.4522 0.4522 0.4522
Pore Size Distribution 147 = 0.49827400 0.49827400 0.49827400
Bubbling Pressure 147 = 0.13099306 0.13099306 0.13099306
Field Capacity 147 = 0.1688 0.1688 0.1688
Wilting Point 147 =  0.0606 0.0606 0.0606
Bulk Density 147 = 1451.55 1451.55 1451.55
Vertical Conductivity 147 = 0.00000500 0.00000500 0.00000500
Thermal Conductivity 147 = 0.6983 0.6983 0.6983
Thermal Capacity 147 = 1406620 1406620 1406620
Mannings n 147 = 0.0300

Soil Description 148 =   74832
Lateral Conductivity 148 = 0.00000900
Exponential Decrease 148 = 0.13912742
Depth Threshold 148 = 0.0500000000
Maximum Infiltration 148 = 0.00000900
Capillary Drive 148 = 0.20160494
Surface Albedo 148 = 0.16
Number of Soil Layers 148 = 3
Porosity 148 = 0.4236 0.4236 0.4236
Pore Size Distribution 148 = 0.51811162 0.51811162 0.51811162
Bubbling Pressure 148 = 0.11808354 0.11808354 0.11808354
Field Capacity 148 = 0.1352 0.1352 0.1352
Wilting Point 148 =  0.0474 0.0474 0.0474
Bulk Density 148 = 1527.53 1527.53 1527.53
Vertical Conductivity 148 = 0.00000845 0.00000845 0.00000845
Thermal Conductivity 148 = 0.7199 0.7199 0.7199
Thermal Capacity 148 = 1320557 1320557 1320557
Mannings n 148 = 0.0300

Soil Description 149 =   74834
Lateral Conductivity 149 = 0.00000600
Exponential Decrease 149 = 0.00000000
Depth Threshold 149 = 0.0500000000
Maximum Infiltration 149 = 0.00000600
Capillary Drive 149 = 0.12695906
Surface Albedo 149 = 0.11
Number of Soil Layers 149 = 3
Porosity 149 = 0.6360 0.6360 0.6360
Pore Size Distribution 149 = 0.49371509 0.49371509 0.49371509
Bubbling Pressure 149 = 0.13395981 0.13395981 0.13395981
Field Capacity 149 = 0.0964 0.0964 0.0964
Wilting Point 149 =  0.0343 0.0343 0.0343
Bulk Density 149 = 964.62 964.62 964.62
Vertical Conductivity 149 = 0.00000600 0.00000600 0.00000600
Thermal Conductivity 149 = 0.6934 0.6934 0.6934
Thermal Capacity 149 = 1426399 1426399 1426399
Mannings n 149 = 0.0300

Soil Description 150 =   74835
Lateral Conductivity 150 = 0.00030000
Exponential Decrease 150 = 0.56829456
Depth Threshold 150 = 0.0500000000
Maximum Infiltration 150 = 0.00030000
Capillary Drive 150 = 0.23394737
Surface Albedo 150 = 0.44
Number of Soil Layers 150 = 3
Porosity 150 = 0.3748 0.3748 0.3748
Pore Size Distribution 150 = 0.58064692 0.58064692 0.58064692
Bubbling Pressure 150 = 0.07738812 0.07738812 0.07738812
Field Capacity 150 = 0.0480 0.0480 0.0480
Wilting Point 150 =  0.0117 0.0117 0.0117
Bulk Density 150 = 1656.84 1656.84 1656.84
Vertical Conductivity 150 = 0.00010721 0.00010721 0.00010721
Thermal Conductivity 150 = 0.7877 0.7877 0.7877
Thermal Capacity 150 = 1049254 1049254 1049254
Mannings n 150 = 0.0300

Soil Description 151 =   74836
Lateral Conductivity 151 = 0.00000900
Exponential Decrease 151 = 0.20699398
Depth Threshold 151 = 0.0500000000
Maximum Infiltration 151 = 0.00030000
Capillary Drive 151 = 0.05684210
Surface Albedo 151 = 0.16
Number of Soil Layers 151 = 3
Porosity 151 = 0.3745 0.3745 0.3745
Pore Size Distribution 151 = 0.57007798 0.57007798 0.57007798
Bubbling Pressure 151 = 0.08426596 0.08426596 0.08426596
Field Capacity 151 = 0.0540 0.0540 0.0540
Wilting Point 151 =  0.0172 0.0172 0.0172
Bulk Density 151 = 1657.57 1657.57 1657.57
Vertical Conductivity 151 = 0.00021606 0.00021606 0.00021606
Thermal Conductivity 151 = 0.7762 0.7762 0.7762
Thermal Capacity 151 = 1095106 1095106 1095106
Mannings n 151 = 0.0300

Soil Description 152 =   74837
Lateral Conductivity 152 = 0.00002800
Exponential Decrease 152 = 0.59612849
Depth Threshold 152 = 0.0500000000
Maximum Infiltration 152 = 0.00009200
Capillary Drive 152 = 0.22184211
Surface Albedo 152 = 0.44
Number of Soil Layers 152 = 3
Porosity 152 = 0.4010 0.4010 0.4010
Pore Size Distribution 152 = 0.57843780 0.57843780 0.57843780
Bubbling Pressure 152 = 0.07882573 0.07882573 0.07882573
Field Capacity 152 = 0.0551 0.0551 0.0551
Wilting Point 152 =  0.0167 0.0167 0.0167
Bulk Density 152 = 1587.37 1587.37 1587.37
Vertical Conductivity 152 = 0.00008442 0.00008442 0.00008442
Thermal Conductivity 152 = 0.7853 0.7853 0.7853
Thermal Capacity 152 = 1058838 1058838 1058838
Mannings n 152 = 0.0300

Soil Description 153 =   74838
Lateral Conductivity 153 = 0.00000900
Exponential Decrease 153 = 0.43826074
Depth Threshold 153 = 0.0500000000
Maximum Infiltration 153 = 0.00002800
Capillary Drive 153 = 0.37335526
Surface Albedo 153 = 0.44
Number of Soil Layers 153 = 3
Porosity 153 = 0.4753 0.4753 0.4753
Pore Size Distribution 153 = 0.45009996 0.45009996 0.45009996
Bubbling Pressure 153 = 0.16234276 0.16234276 0.16234276
Field Capacity 153 = 0.2078 0.2078 0.2078
Wilting Point 153 =  0.1036 0.1036 0.1036
Bulk Density 153 = 1390.46 1390.46 1390.46
Vertical Conductivity 153 = 0.00001346 0.00001346 0.00001346
Thermal Conductivity 153 = 0.6461 0.6461 0.6461
Thermal Capacity 153 = 1615618 1615618 1615618
Mannings n 153 = 0.0300

Soil Description 154 =   74839
Lateral Conductivity 154 = 0.00000900
Exponential Decrease 154 = 0.17063392
Depth Threshold 154 = 0.0500000000
Maximum Infiltration 154 = 0.00000900
Capillary Drive 154 = 0.25901316
Surface Albedo 154 = 0.16
Number of Soil Layers 154 = 3
Porosity 154 = 0.4319 0.4319 0.4319
Pore Size Distribution 154 = 0.47823755 0.47823755 0.47823755
Bubbling Pressure 154 = 0.14403196 0.14403196 0.14403196
Field Capacity 154 = 0.1817 0.1817 0.1817
Wilting Point 154 =  0.0737 0.0737 0.0737
Bulk Density 154 = 1505.46 1505.46 1505.46
Vertical Conductivity 154 = 0.00000473 0.00000473 0.00000473
Thermal Conductivity 154 = 0.6766 0.6766 0.6766
Thermal Capacity 154 = 1493546 1493546 1493546
Mannings n 154 = 0.0300

Soil Description 155 =   74840
Lateral Conductivity 155 = 0.00000900
Exponential Decrease 155 = 1.14041396
Depth Threshold 155 = 0.0500000000
Maximum Infiltration 155 = 0.00000900
Capillary Drive 155 = 0.22916192
Surface Albedo 155 = 0.23
Number of Soil Layers 155 = 3
Porosity 155 = 0.4886 0.4886 0.4886
Pore Size Distribution 155 = 0.48280329 0.48280329 0.48280329
Bubbling Pressure 155 = 0.14106077 0.14106077 0.14106077
Field Capacity 155 = 0.1613 0.1613 0.1613
Wilting Point 155 =  0.0620 0.0620 0.0620
Bulk Density 155 = 1355.17 1355.17 1355.17
Vertical Conductivity 155 = 0.00000615 0.00000615 0.00000615
Thermal Conductivity 155 = 0.6816 0.6816 0.6816
Thermal Capacity 155 = 1473738 1473738 1473738
Mannings n 155 = 0.0300

Soil Description 156 =   74841
Lateral Conductivity 156 = 0.00002800
Exponential Decrease 156 = 0.11082437
Depth Threshold 156 = 0.0500000000
Maximum Infiltration 156 = 0.00009200
Capillary Drive 156 = 0.09447368
Surface Albedo 156 = 0.37
Number of Soil Layers 156 = 3
Porosity 156 = 0.3910 0.3910 0.3910
Pore Size Distribution 156 = 0.56855632 0.56855632 0.56855632
Bubbling Pressure 156 = 0.08525619 0.08525619 0.08525619
Field Capacity 156 = 0.0485 0.0485 0.0485
Wilting Point 156 =  0.0164 0.0164 0.0164
Bulk Density 156 = 1613.75 1613.75 1613.75
Vertical Conductivity 156 = 0.00003786 0.00003786 0.00003786
Thermal Conductivity 156 = 0.7746 0.7746 0.7746
Thermal Capacity 156 = 1101708 1101708 1101708
Mannings n 156 = 0.0300

Soil Description 157 =   74843
Lateral Conductivity 157 = 0.00000900
Exponential Decrease 157 = 0.76057420
Depth Threshold 157 = 0.0500000000
Maximum Infiltration 157 = 0.00002800
Capillary Drive 157 = 0.53289474
Surface Albedo 157 = 0.30
Number of Soil Layers 157 = 3
Porosity 157 = 0.5568 0.5568 0.5568
Pore Size Distribution 157 = 0.55324306 0.55324306 0.55324306
Bubbling Pressure 157 = 0.09522144 0.09522144 0.09522144
Field Capacity 157 = 0.1645 0.1645 0.1645
Wilting Point 157 =  0.0655 0.0655 0.0655
Bulk Density 157 = 1174.54 1174.54 1174.54
Vertical Conductivity 157 = 0.00002138 0.00002138 0.00002138
Thermal Conductivity 157 = 0.7580 0.7580 0.7580
Thermal Capacity 157 = 1168143 1168143 1168143
Mannings n 157 = 0.0300

Soil Description 158 =   74844
Lateral Conductivity 158 = 0.00000900
Exponential Decrease 158 = 0.76057420
Depth Threshold 158 = 0.0500000000
Maximum Infiltration 158 = 0.00002800
Capillary Drive 158 = 0.53289474
Surface Albedo 158 = 0.30
Number of Soil Layers 158 = 3
Porosity 158 = 0.5568 0.5568 0.5568
Pore Size Distribution 158 = 0.55324306 0.55324306 0.55324306
Bubbling Pressure 158 = 0.09522144 0.09522144 0.09522144
Field Capacity 158 = 0.1645 0.1645 0.1645
Wilting Point 158 =  0.0655 0.0655 0.0655
Bulk Density 158 = 1174.54 1174.54 1174.54
Vertical Conductivity 158 = 0.00002138 0.00002138 0.00002138
Thermal Conductivity 158 = 0.7580 0.7580 0.7580
Thermal Capacity 158 = 1168143 1168143 1168143
Mannings n 158 = 0.0300

Soil Description 159 =   74845
Lateral Conductivity 159 = 0.00000900
Exponential Decrease 159 = 0.16340922
Depth Threshold 159 = 0.0500000000
Maximum Infiltration 159 = 0.00000900
Capillary Drive 159 = 0.30743421
Surface Albedo 159 = 0.37
Number of Soil Layers 159 = 3
Porosity 159 = 0.5189 0.5189 0.5189
Pore Size Distribution 159 = 0.47102751 0.47102751 0.47102751
Bubbling Pressure 159 = 0.14872396 0.14872396 0.14872396
Field Capacity 159 = 0.1931 0.1931 0.1931
Wilting Point 159 =  0.0705 0.0705 0.0705
Bulk Density 159 = 1275.00 1275.00 1275.00
Vertical Conductivity 159 = 0.00000560 0.00000560 0.00000560
Thermal Conductivity 159 = 0.6688 0.6688 0.6688
Thermal Capacity 159 = 1524826 1524826 1524826
Mannings n 159 = 0.0300

Soil Description 160 =   74846
Lateral Conductivity 160 = 0.00000900
Exponential Decrease 160 = 0.00000000
Depth Threshold 160 = 0.0500000000
Maximum Infiltration 160 = 0.00000900
Capillary Drive 160 = 0.58828947
Surface Albedo 160 = 0.30
Number of Soil Layers 160 = 3
Porosity 160 = 0.6151 0.6151 0.6151
Pore Size Distribution 160 = 0.30050096 0.30050096 0.30050096
Bubbling Pressure 160 = 0.25969569 0.25969569 0.25969569
Field Capacity 160 = 0.2496 0.2496 0.2496
Wilting Point 160 =  0.0949 0.0949 0.0949
Bulk Density 160 = 1020.00 1020.00 1020.00
Vertical Conductivity 160 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 160 = 0.4838 0.4838 0.4838
Thermal Capacity 160 = 2264638 2264638 2264638
Mannings n 160 = 0.0300

Soil Description 161 =   74847
Lateral Conductivity 161 = 0.00002800
Exponential Decrease 161 = 0.32749496
Depth Threshold 161 = 0.0500000000
Maximum Infiltration 161 = 0.00030000
Capillary Drive 161 = 0.04881579
Surface Albedo 161 = 0.16
Number of Soil Layers 161 = 3
Porosity 161 = 0.3828 0.3828 0.3828
Pore Size Distribution 161 = 0.56998995 0.56998995 0.56998995
Bubbling Pressure 161 = 0.08432325 0.08432325 0.08432325
Field Capacity 161 = 0.0425 0.0425 0.0425
Wilting Point 161 =  0.0164 0.0164 0.0164
Bulk Density 161 = 1635.59 1635.59 1635.59
Vertical Conductivity 161 = 0.00020874 0.00020874 0.00020874
Thermal Conductivity 161 = 0.7761 0.7761 0.7761
Thermal Capacity 161 = 1095488 1095488 1095488
Mannings n 161 = 0.0300

Soil Description 162 =   74848
Lateral Conductivity 162 = 0.00000900
Exponential Decrease 162 = 0.52914352
Depth Threshold 162 = 0.0500000000
Maximum Infiltration 162 = 0.00002800
Capillary Drive 162 = 0.45105263
Surface Albedo 162 = 0.23
Number of Soil Layers 162 = 3
Porosity 162 = 0.4710 0.4710 0.4710
Pore Size Distribution 162 = 0.40205067 0.40205067 0.40205067
Bubbling Pressure 162 = 0.19361128 0.19361128 0.19361128
Field Capacity 162 = 0.2580 0.2580 0.2580
Wilting Point 162 =  0.1237 0.1237 0.1237
Bulk Density 162 = 1401.97 1401.97 1401.97
Vertical Conductivity 162 = 0.00001040 0.00001040 0.00001040
Thermal Conductivity 162 = 0.5940 0.5940 0.5940
Thermal Capacity 162 = 1824075 1824075 1824075
Mannings n 162 = 0.0300

Soil Description 163 =   74849
Lateral Conductivity 163 = 0.00000900
Exponential Decrease 163 = 0.52160483
Depth Threshold 163 = 0.0500000000
Maximum Infiltration 163 = 0.00009200
Capillary Drive 163 = 0.36467105
Surface Albedo 163 = 0.30
Number of Soil Layers 163 = 3
Porosity 163 = 0.4547 0.4547 0.4547
Pore Size Distribution 163 = 0.52712534 0.52712534 0.52712534
Bubbling Pressure 163 = 0.11221778 0.11221778 0.11221778
Field Capacity 163 = 0.1725 0.1725 0.1725
Wilting Point 163 =  0.0605 0.0605 0.0605
Bulk Density 163 = 1445.00 1445.00 1445.00
Vertical Conductivity 163 = 0.00003177 0.00003177 0.00003177
Thermal Conductivity 163 = 0.7296 0.7296 0.7296
Thermal Capacity 163 = 1281452 1281452 1281452
Mannings n 163 = 0.0300

Soil Description 164 =   74850
Lateral Conductivity 164 = 0.00002800
Exponential Decrease 164 = 0.61596273
Depth Threshold 164 = 0.0500000000
Maximum Infiltration 164 = 0.00009200
Capillary Drive 164 = 0.35519737
Surface Albedo 164 = 0.30
Number of Soil Layers 164 = 3
Porosity 164 = 0.4100 0.4100 0.4100
Pore Size Distribution 164 = 0.57892208 0.57892208 0.57892208
Bubbling Pressure 164 = 0.07851057 0.07851057 0.07851057
Field Capacity 164 = 0.0899 0.0899 0.0899
Wilting Point 164 =  0.0240 0.0240 0.0240
Bulk Density 164 = 1563.49 1563.49 1563.49
Vertical Conductivity 164 = 0.00008147 0.00008147 0.00008147
Thermal Conductivity 164 = 0.7858 0.7858 0.7858
Thermal Capacity 164 = 1056737 1056737 1056737
Mannings n 164 = 0.0300

Soil Description 165 =   74854
Lateral Conductivity 165 = 0.00000900
Exponential Decrease 165 = 0.00000000
Depth Threshold 165 = 0.0500000000
Maximum Infiltration 165 = 0.00000900
Capillary Drive 165 = 0.21848684
Surface Albedo 165 = 0.23
Number of Soil Layers 165 = 3
Porosity 165 = 0.5510 0.5510 0.5510
Pore Size Distribution 165 = 0.49642050 0.49642050 0.49642050
Bubbling Pressure 165 = 0.13219924 0.13219924 0.13219924
Field Capacity 165 = 0.1346 0.1346 0.1346
Wilting Point 165 =  0.0402 0.0402 0.0402
Bulk Density 165 = 1189.80 1189.80 1189.80
Vertical Conductivity 165 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 165 = 0.6963 0.6963 0.6963
Thermal Capacity 165 = 1414662 1414662 1414662
Mannings n 165 = 0.0300

Soil Description 166 =   74855
Lateral Conductivity 166 = 0.00000900
Exponential Decrease 166 = 0.55626726
Depth Threshold 166 = 0.0500000000
Maximum Infiltration 166 = 0.00000900
Capillary Drive 166 = 0.46657895
Surface Albedo 166 = 0.30
Number of Soil Layers 166 = 3
Porosity 166 = 0.4911 0.4911 0.4911
Pore Size Distribution 166 = 0.20430615 0.20430615 0.20430615
Bubbling Pressure 166 = 0.32229535 0.32229535 0.32229535
Field Capacity 166 = 0.3076 0.3076 0.3076
Wilting Point 166 =  0.2200 0.2200 0.2200
Bulk Density 166 = 1348.68 1348.68 1348.68
Vertical Conductivity 166 = 0.00000326 0.00000326 0.00000326
Thermal Conductivity 166 = 0.3795 0.3795 0.3795
Thermal Capacity 166 = 2681969 2681969 2681969
Mannings n 166 = 0.0300

Soil Description 167 =   74856
Lateral Conductivity 167 = 0.00000900
Exponential Decrease 167 = 0.42298260
Depth Threshold 167 = 0.0500000000
Maximum Infiltration 167 = 0.00000900
Capillary Drive 167 = 0.46203947
Surface Albedo 167 = 0.30
Number of Soil Layers 167 = 3
Porosity 167 = 0.4786 0.4786 0.4786
Pore Size Distribution 167 = 0.25885577 0.25885577 0.25885577
Bubbling Pressure 167 = 0.28679668 0.28679668 0.28679668
Field Capacity 167 = 0.3029 0.3029 0.3029
Wilting Point 167 =  0.2164 0.2164 0.2164
Bulk Density 167 = 1381.78 1381.78 1381.78
Vertical Conductivity 167 = 0.00000474 0.00000474 0.00000474
Thermal Conductivity 167 = 0.4387 0.4387 0.4387
Thermal Capacity 167 = 2445311 2445311 2445311
Mannings n 167 = 0.0300

Soil Description 168 =   74857
Lateral Conductivity 168 = 0.00000900
Exponential Decrease 168 = 0.47914872
Depth Threshold 168 = 0.0500000000
Maximum Infiltration 168 = 0.00000900
Capillary Drive 168 = 0.46315789
Surface Albedo 168 = 0.30
Number of Soil Layers 168 = 3
Porosity 168 = 0.4917 0.4917 0.4917
Pore Size Distribution 168 = 0.22939257 0.22939257 0.22939257
Bubbling Pressure 168 = 0.30597013 0.30597013 0.30597013
Field Capacity 168 = 0.3019 0.3019 0.3019
Wilting Point 168 =  0.2065 0.2065 0.2065
Bulk Density 168 = 1346.97 1346.97 1346.97
Vertical Conductivity 168 = 0.00000395 0.00000395 0.00000395
Thermal Conductivity 168 = 0.4067 0.4067 0.4067
Thermal Capacity 168 = 2573134 2573134 2573134
Mannings n 168 = 0.0300

Soil Description 169 =   74858
Lateral Conductivity 169 = 0.00002800
Exponential Decrease 169 = 0.34278886
Depth Threshold 169 = 0.0500000000
Maximum Infiltration 169 = 0.00030000
Capillary Drive 169 = 0.05671053
Surface Albedo 169 = 0.16
Number of Soil Layers 169 = 3
Porosity 169 = 0.4007 0.4007 0.4007
Pore Size Distribution 169 = 0.56374600 0.56374600 0.56374600
Bubbling Pressure 169 = 0.08838655 0.08838655 0.08838655
Field Capacity 169 = 0.0482 0.0482 0.0482
Wilting Point 169 =  0.0180 0.0180 0.0180
Bulk Density 169 = 1588.22 1588.22 1588.22
Vertical Conductivity 169 = 0.00019084 0.00019084 0.00019084
Thermal Conductivity 169 = 0.7694 0.7694 0.7694
Thermal Capacity 169 = 1122577 1122577 1122577
Mannings n 169 = 0.0300

Soil Description 170 =   74859
Lateral Conductivity 170 = 0.00000900
Exponential Decrease 170 = 0.00000000
Depth Threshold 170 = 0.0500000000
Maximum Infiltration 170 = 0.00000900
Capillary Drive 170 = 0.18310345
Surface Albedo 170 = 0.30
Number of Soil Layers 170 = 3
Porosity 170 = 0.5976 0.5976 0.5976
Pore Size Distribution 170 = 0.53181723 0.53181723 0.53181723
Bubbling Pressure 170 = 0.10916449 0.10916449 0.10916449
Field Capacity 170 = 0.1058 0.1058 0.1058
Wilting Point 170 =  0.0541 0.0541 0.0541
Bulk Density 170 = 1066.38 1066.38 1066.38
Vertical Conductivity 170 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 170 = 0.7347 0.7347 0.7347
Thermal Capacity 170 = 1261097 1261097 1261097
Mannings n 170 = 0.0300

Soil Description 171 =   74860
Lateral Conductivity 171 = 0.00000900
Exponential Decrease 171 = 0.00000000
Depth Threshold 171 = 0.0500000000
Maximum Infiltration 171 = 0.00000900
Capillary Drive 171 = 0.18310345
Surface Albedo 171 = 0.30
Number of Soil Layers 171 = 3
Porosity 171 = 0.5976 0.5976 0.5976
Pore Size Distribution 171 = 0.53181723 0.53181723 0.53181723
Bubbling Pressure 171 = 0.10916449 0.10916449 0.10916449
Field Capacity 171 = 0.1058 0.1058 0.1058
Wilting Point 171 =  0.0541 0.0541 0.0541
Bulk Density 171 = 1066.38 1066.38 1066.38
Vertical Conductivity 171 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 171 = 0.7347 0.7347 0.7347
Thermal Capacity 171 = 1261097 1261097 1261097
Mannings n 171 = 0.0300

Soil Description 172 =   74862
Lateral Conductivity 172 = 0.00000900
Exponential Decrease 172 = 0.17232696
Depth Threshold 172 = 0.0500000000
Maximum Infiltration 172 = 0.00000900
Capillary Drive 172 = 0.08710526
Surface Albedo 172 = 0.37
Number of Soil Layers 172 = 3
Porosity 172 = 0.3623 0.3623 0.3623
Pore Size Distribution 172 = 0.52183257 0.52183257 0.52183257
Bubbling Pressure 172 = 0.11566210 0.11566210 0.11566210
Field Capacity 172 = 0.0946 0.0946 0.0946
Wilting Point 172 =  0.0339 0.0339 0.0339
Bulk Density 172 = 1690.00 1690.00 1690.00
Vertical Conductivity 172 = 0.00000455 0.00000455 0.00000455
Thermal Conductivity 172 = 0.7239 0.7239 0.7239
Thermal Capacity 172 = 1304414 1304414 1304414
Mannings n 172 = 0.0300

Soil Description 173 =   74863
Lateral Conductivity 173 = 0.00000900
Exponential Decrease 173 = 0.17232696
Depth Threshold 173 = 0.0500000000
Maximum Infiltration 173 = 0.00000900
Capillary Drive 173 = 0.08710526
Surface Albedo 173 = 0.37
Number of Soil Layers 173 = 3
Porosity 173 = 0.3623 0.3623 0.3623
Pore Size Distribution 173 = 0.52183257 0.52183257 0.52183257
Bubbling Pressure 173 = 0.11566210 0.11566210 0.11566210
Field Capacity 173 = 0.0946 0.0946 0.0946
Wilting Point 173 =  0.0339 0.0339 0.0339
Bulk Density 173 = 1690.00 1690.00 1690.00
Vertical Conductivity 173 = 0.00000455 0.00000455 0.00000455
Thermal Conductivity 173 = 0.7239 0.7239 0.7239
Thermal Capacity 173 = 1304414 1304414 1304414
Mannings n 173 = 0.0300

Soil Description 174 =   74864
Lateral Conductivity 174 = 0.00002800
Exponential Decrease 174 = 0.34132351
Depth Threshold 174 = 0.0500000000
Maximum Infiltration 174 = 0.00030000
Capillary Drive 174 = 0.36763158
Surface Albedo 174 = 0.16
Number of Soil Layers 174 = 3
Porosity 174 = 0.4275 0.4275 0.4275
Pore Size Distribution 174 = 0.57790488 0.57790488 0.57790488
Bubbling Pressure 174 = 0.07917253 0.07917253 0.07917253
Field Capacity 174 = 0.1064 0.1064 0.1064
Wilting Point 174 =  0.0358 0.0358 0.0358
Bulk Density 174 = 1517.17 1517.17 1517.17
Vertical Conductivity 174 = 0.00019242 0.00019242 0.00019242
Thermal Conductivity 174 = 0.7847 0.7847 0.7847
Thermal Capacity 174 = 1061150 1061150 1061150
Mannings n 174 = 0.0300

Soil Description 175 =   74865
Lateral Conductivity 175 = 0.00000900
Exponential Decrease 175 = 0.69319167
Depth Threshold 175 = 0.0500000000
Maximum Infiltration 175 = 0.00002800
Capillary Drive 175 = 0.21328947
Surface Albedo 175 = 0.30
Number of Soil Layers 175 = 3
Porosity 175 = 0.5583 0.5583 0.5583
Pore Size Distribution 175 = 0.52747935 0.52747935 0.52747935
Bubbling Pressure 175 = 0.11198740 0.11198740 0.11198740
Field Capacity 175 = 0.0993 0.0993 0.0993
Wilting Point 175 =  0.0324 0.0324 0.0324
Bulk Density 175 = 1170.53 1170.53 1170.53
Vertical Conductivity 175 = 0.00002325 0.00002325 0.00002325
Thermal Conductivity 175 = 0.7300 0.7300 0.7300
Thermal Capacity 175 = 1279916 1279916 1279916
Mannings n 175 = 0.0300

Soil Description 176 =   74866
Lateral Conductivity 176 = 0.00000900
Exponential Decrease 176 = 0.20456140
Depth Threshold 176 = 0.0500000000
Maximum Infiltration 176 = 0.00030000
Capillary Drive 176 = 0.04572368
Surface Albedo 176 = 0.37
Number of Soil Layers 176 = 3
Porosity 176 = 0.3895 0.3895 0.3895
Pore Size Distribution 176 = 0.56722842 0.56722842 0.56722842
Bubbling Pressure 176 = 0.08612033 0.08612033 0.08612033
Field Capacity 176 = 0.0495 0.0495 0.0495
Wilting Point 176 =  0.0166 0.0166 0.0166
Bulk Density 176 = 1617.83 1617.83 1617.83
Vertical Conductivity 176 = 0.00022438 0.00022438 0.00022438
Thermal Conductivity 176 = 0.7731 0.7731 0.7731
Thermal Capacity 176 = 1107469 1107469 1107469
Mannings n 176 = 0.0300

Soil Description 177 =   74869
Lateral Conductivity 177 = 0.00002800
Exponential Decrease 177 = 0.32079267
Depth Threshold 177 = 0.0500000000
Maximum Infiltration 177 = 0.00030000
Capillary Drive 177 = 0.04881579
Surface Albedo 177 = 0.16
Number of Soil Layers 177 = 3
Porosity 177 = 0.3801 0.3801 0.3801
Pore Size Distribution 177 = 0.56893994 0.56893994 0.56893994
Bubbling Pressure 177 = 0.08500655 0.08500655 0.08500655
Field Capacity 177 = 0.0453 0.0453 0.0453
Wilting Point 177 =  0.0161 0.0161 0.0161
Bulk Density 177 = 1642.76 1642.76 1642.76
Vertical Conductivity 177 = 0.00021768 0.00021768 0.00021768
Thermal Conductivity 177 = 0.7750 0.7750 0.7750
Thermal Capacity 177 = 1100044 1100044 1100044
Mannings n 177 = 0.0300

Soil Description 178 =   74870
Lateral Conductivity 178 = 0.00000900
Exponential Decrease 178 = 0.17877777
Depth Threshold 178 = 0.0500000000
Maximum Infiltration 178 = 0.00000900
Capillary Drive 178 = 0.09263158
Surface Albedo 178 = 0.37
Number of Soil Layers 178 = 3
Porosity 178 = 0.3723 0.3723 0.3723
Pore Size Distribution 178 = 0.49112332 0.49112332 0.49112332
Bubbling Pressure 178 = 0.13564643 0.13564643 0.13564643
Field Capacity 178 = 0.1193 0.1193 0.1193
Wilting Point 178 =  0.0419 0.0419 0.0419
Bulk Density 178 = 1663.42 1663.42 1663.42
Vertical Conductivity 178 = 0.00000396 0.00000396 0.00000396
Thermal Conductivity 178 = 0.6906 0.6906 0.6906
Thermal Capacity 178 = 1437643 1437643 1437643
Mannings n 178 = 0.0300

Soil Description 179 =   74871
Lateral Conductivity 179 = 0.00000900
Exponential Decrease 179 = 0.16703061
Depth Threshold 179 = 0.0500000000
Maximum Infiltration 179 = 0.00000900
Capillary Drive 179 = 0.12921053
Surface Albedo 179 = 0.30
Number of Soil Layers 179 = 3
Porosity 179 = 0.4228 0.4228 0.4228
Pore Size Distribution 179 = 0.49821825 0.49821825 0.49821825
Bubbling Pressure 179 = 0.13102934 0.13102934 0.13102934
Field Capacity 179 = 0.1221 0.1221 0.1221
Wilting Point 179 =  0.0397 0.0397 0.0397
Bulk Density 179 = 1529.47 1529.47 1529.47
Vertical Conductivity 179 = 0.00000514 0.00000514 0.00000514
Thermal Conductivity 179 = 0.6983 0.6983 0.6983
Thermal Capacity 179 = 1406862 1406862 1406862
Mannings n 179 = 0.0300

Soil Description 180 =   74872
Lateral Conductivity 180 = 0.00000900
Exponential Decrease 180 = 0.00000000
Depth Threshold 180 = 0.0500000000
Maximum Infiltration 180 = 0.00000900
Capillary Drive 180 = 0.19591549
Surface Albedo 180 = 0.23
Number of Soil Layers 180 = 3
Porosity 180 = 0.5454 0.5454 0.5454
Pore Size Distribution 180 = 0.50787591 0.50787591 0.50787591
Bubbling Pressure 180 = 0.12474453 0.12474453 0.12474453
Field Capacity 180 = 0.1385 0.1385 0.1385
Wilting Point 180 =  0.0498 0.0498 0.0498
Bulk Density 180 = 1204.65 1204.65 1204.65
Vertical Conductivity 180 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 180 = 0.7088 0.7088 0.7088
Thermal Capacity 180 = 1364964 1364964 1364964
Mannings n 180 = 0.0300

Soil Description 181 =   74873
Lateral Conductivity 181 = 0.00000663
Exponential Decrease 181 = 0.00000000
Depth Threshold 181 = 0.0500000000
Maximum Infiltration 181 = 0.00000663
Capillary Drive 181 = 0.14435878
Surface Albedo 181 = 0.17
Number of Soil Layers 181 = 3
Porosity 181 = 0.6650 0.6650 0.6650
Pore Size Distribution 181 = 0.50787591 0.50787591 0.50787591
Bubbling Pressure 181 = 0.12474453 0.12474453 0.12474453
Field Capacity 181 = 0.1021 0.1021 0.1021
Wilting Point 181 =  0.0367 0.0367 0.0367
Bulk Density 181 = 887.64 887.64 887.64
Vertical Conductivity 181 = 0.00000663 0.00000663 0.00000663
Thermal Conductivity 181 = 0.7088 0.7088 0.7088
Thermal Capacity 181 = 1364964 1364964 1364964
Mannings n 181 = 0.0300

Soil Description 182 =   74874
Lateral Conductivity 182 = 0.00002800
Exponential Decrease 182 = 0.31113006
Depth Threshold 182 = 0.0500000000
Maximum Infiltration 182 = 0.00030000
Capillary Drive 182 = 0.10750000
Surface Albedo 182 = 0.09
Number of Soil Layers 182 = 3
Porosity 182 = 0.3550 0.3550 0.3550
Pore Size Distribution 182 = 0.56831935 0.56831935 0.56831935
Bubbling Pressure 182 = 0.08541040 0.08541040 0.08541040
Field Capacity 182 = 0.0520 0.0520 0.0520
Wilting Point 182 =  0.0203 0.0203 0.0203
Bulk Density 182 = 1709.21 1709.21 1709.21
Vertical Conductivity 182 = 0.00023200 0.00023200 0.00023200
Thermal Conductivity 182 = 0.7743 0.7743 0.7743
Thermal Capacity 182 = 1102736 1102736 1102736
Mannings n 182 = 0.0300

Soil Description 183 =   74877
Lateral Conductivity 183 = 0.00002800
Exponential Decrease 183 = 0.74077247
Depth Threshold 183 = 0.0500000000
Maximum Infiltration 183 = 0.00002800
Capillary Drive 183 = 0.45210526
Surface Albedo 183 = 0.23
Number of Soil Layers 183 = 3
Porosity 183 = 0.4665 0.4665 0.4665
Pore Size Distribution 183 = 0.55907470 0.55907470 0.55907470
Bubbling Pressure 183 = 0.09142644 0.09142644 0.09142644
Field Capacity 183 = 0.1311 0.1311 0.1311
Wilting Point 183 =  0.0533 0.0533 0.0533
Bulk Density 183 = 1413.82 1413.82 1413.82
Vertical Conductivity 183 = 0.00002187 0.00002187 0.00002187
Thermal Conductivity 183 = 0.7643 0.7643 0.7643
Thermal Capacity 183 = 1142843 1142843 1142843
Mannings n 183 = 0.0300

Soil Description 184 =   74878
Lateral Conductivity 184 = 0.00002800
Exponential Decrease 184 = 0.74077247
Depth Threshold 184 = 0.0500000000
Maximum Infiltration 184 = 0.00002800
Capillary Drive 184 = 0.45210526
Surface Albedo 184 = 0.23
Number of Soil Layers 184 = 3
Porosity 184 = 0.4665 0.4665 0.4665
Pore Size Distribution 184 = 0.55907470 0.55907470 0.55907470
Bubbling Pressure 184 = 0.09142644 0.09142644 0.09142644
Field Capacity 184 = 0.1311 0.1311 0.1311
Wilting Point 184 =  0.0533 0.0533 0.0533
Bulk Density 184 = 1413.82 1413.82 1413.82
Vertical Conductivity 184 = 0.00002187 0.00002187 0.00002187
Thermal Conductivity 184 = 0.7643 0.7643 0.7643
Thermal Capacity 184 = 1142843 1142843 1142843
Mannings n 184 = 0.0300

Soil Description 185 =   74879
Lateral Conductivity 185 = 0.00000900
Exponential Decrease 185 = 1.34388136
Depth Threshold 185 = 0.0500000000
Maximum Infiltration 185 = 0.00000900
Capillary Drive 185 = 0.49223684
Surface Albedo 185 = 0.23
Number of Soil Layers 185 = 3
Porosity 185 = 0.5003 0.5003 0.5003
Pore Size Distribution 185 = 0.43330928 0.43330928 0.43330928
Bubbling Pressure 185 = 0.17326945 0.17326945 0.17326945
Field Capacity 185 = 0.2413 0.2413 0.2413
Wilting Point 185 =  0.1365 0.1365 0.1365
Bulk Density 185 = 1324.21 1324.21 1324.21
Vertical Conductivity 185 = 0.00000489 0.00000489 0.00000489
Thermal Conductivity 185 = 0.6279 0.6279 0.6279
Thermal Capacity 185 = 1688463 1688463 1688463
Mannings n 185 = 0.0300

Soil Description 186 =   74882
Lateral Conductivity 186 = 0.00001976
Exponential Decrease 186 = 0.00000000
Depth Threshold 186 = 0.0500000000
Maximum Infiltration 186 = 0.00001976
Capillary Drive 186 = 0.09674060
Surface Albedo 186 = 0.31
Number of Soil Layers 186 = 3
Porosity 186 = 0.6092 0.6092 0.6092
Pore Size Distribution 186 = 0.56705391 0.56705391 0.56705391
Bubbling Pressure 186 = 0.08623390 0.08623390 0.08623390
Field Capacity 186 = 0.0508 0.0508 0.0508
Wilting Point 186 =  0.0198 0.0198 0.0198
Bulk Density 186 = 1035.56 1035.56 1035.56
Vertical Conductivity 186 = 0.00001976 0.00001976 0.00001976
Thermal Conductivity 186 = 0.7729 0.7729 0.7729
Thermal Capacity 186 = 1108226 1108226 1108226
Mannings n 186 = 0.0300

Soil Description 187 =   74883
Lateral Conductivity 187 = 0.00000900
Exponential Decrease 187 = 0.85088378
Depth Threshold 187 = 0.0500000000
Maximum Infiltration 187 = 0.00002300
Capillary Drive 187 = 0.51000000
Surface Albedo 187 = 0.30
Number of Soil Layers 187 = 3
Porosity 187 = 0.5258 0.5258 0.5258
Pore Size Distribution 187 = 0.55080622 0.55080622 0.55080622
Bubbling Pressure 187 = 0.09680723 0.09680723 0.09680723
Field Capacity 187 = 0.1510 0.1510 0.1510
Wilting Point 187 =  0.0610 0.0610 0.0610
Bulk Density 187 = 1256.58 1256.58 1256.58
Vertical Conductivity 187 = 0.00001950 0.00001950 0.00001950
Thermal Conductivity 187 = 0.7553 0.7553 0.7553
Thermal Capacity 187 = 1178715 1178715 1178715
Mannings n 187 = 0.0300

Soil Description 188 =   74884
Lateral Conductivity 188 = 0.00000900
Exponential Decrease 188 = 0.16955742
Depth Threshold 188 = 0.0500000000
Maximum Infiltration 188 = 0.00000900
Capillary Drive 188 = 0.13934210
Surface Albedo 188 = 0.23
Number of Soil Layers 188 = 3
Porosity 188 = 0.4182 0.4182 0.4182
Pore Size Distribution 188 = 0.50872767 0.50872767 0.50872767
Bubbling Pressure 188 = 0.12419024 0.12419024 0.12419024
Field Capacity 188 = 0.1195 0.1195 0.1195
Wilting Point 188 =  0.0398 0.0398 0.0398
Bulk Density 188 = 1541.71 1541.71 1541.71
Vertical Conductivity 188 = 0.00000484 0.00000484 0.00000484
Thermal Conductivity 188 = 0.7097 0.7097 0.7097
Thermal Capacity 188 = 1361268 1361268 1361268
Mannings n 188 = 0.0300

Soil Description 189 =   74885
Lateral Conductivity 189 = 0.00000900
Exponential Decrease 189 = 0.16257409
Depth Threshold 189 = 0.0500000000
Maximum Infiltration 189 = 0.00000900
Capillary Drive 189 = 0.14032895
Surface Albedo 189 = 0.23
Number of Soil Layers 189 = 3
Porosity 189 = 0.4488 0.4488 0.4488
Pore Size Distribution 189 = 0.47142426 0.47142426 0.47142426
Bubbling Pressure 189 = 0.14846578 0.14846578 0.14846578
Field Capacity 189 = 0.1170 0.1170 0.1170
Wilting Point 189 =  0.0389 0.0389 0.0389
Bulk Density 189 = 1460.79 1460.79 1460.79
Vertical Conductivity 189 = 0.00000572 0.00000572 0.00000572
Thermal Conductivity 189 = 0.6692 0.6692 0.6692
Thermal Capacity 189 = 1523105 1523105 1523105
Mannings n 189 = 0.0300

Soil Description 190 =   74886
Lateral Conductivity 190 = 0.00000900
Exponential Decrease 190 = 0.96511574
Depth Threshold 190 = 0.0500000000
Maximum Infiltration 190 = 0.00000900
Capillary Drive 190 = 0.14820015
Surface Albedo 190 = 0.27
Number of Soil Layers 190 = 3
Porosity 190 = 0.4723 0.4723 0.4723
Pore Size Distribution 190 = 0.50430619 0.50430619 0.50430619
Bubbling Pressure 190 = 0.12706756 0.12706756 0.12706756
Field Capacity 190 = 0.1198 0.1198 0.1198
Wilting Point 190 =  0.0394 0.0394 0.0394
Bulk Density 190 = 1398.44 1398.44 1398.44
Vertical Conductivity 190 = 0.00000607 0.00000607 0.00000607
Thermal Conductivity 190 = 0.7049 0.7049 0.7049
Thermal Capacity 190 = 1380450 1380450 1380450
Mannings n 190 = 0.0300

Soil Description 191 =   74887
Lateral Conductivity 191 = 0.00000900
Exponential Decrease 191 = 0.00000000
Depth Threshold 191 = 0.0500000000
Maximum Infiltration 191 = 0.00000900
Capillary Drive 191 = 0.54171053
Surface Albedo 191 = 0.23
Number of Soil Layers 191 = 3
Porosity 191 = 0.5628 0.5628 0.5628
Pore Size Distribution 191 = 0.53419492 0.53419492 0.53419492
Bubbling Pressure 191 = 0.10761719 0.10761719 0.10761719
Field Capacity 191 = 0.1823 0.1823 0.1823
Wilting Point 191 =  0.0597 0.0597 0.0597
Bulk Density 191 = 1158.62 1158.62 1158.62
Vertical Conductivity 191 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 191 = 0.7373 0.7373 0.7373
Thermal Capacity 191 = 1250781 1250781 1250781
Mannings n 191 = 0.0300

Soil Description 192 =   74890
Lateral Conductivity 192 = 0.00000900
Exponential Decrease 192 = 3.65241070
Depth Threshold 192 = 0.0500000000
Maximum Infiltration 192 = 0.00000900
Capillary Drive 192 = 0.43592105
Surface Albedo 192 = 0.30
Number of Soil Layers 192 = 3
Porosity 192 = 0.4724 0.4724 0.4724
Pore Size Distribution 192 = 0.26173148 0.26173148 0.26173148
Bubbling Pressure 192 = 0.28492528 0.28492528 0.28492528
Field Capacity 192 = 0.2922 0.2922 0.2922
Wilting Point 192 =  0.1489 0.1489 0.1489
Bulk Density 192 = 1398.16 1398.16 1398.16
Vertical Conductivity 192 = 0.00000359 0.00000359 0.00000359
Thermal Conductivity 192 = 0.4418 0.4418 0.4418
Thermal Capacity 192 = 2432835 2432835 2432835
Mannings n 192 = 0.0300

Soil Description 193 =   74896
Lateral Conductivity 193 = 0.00002800
Exponential Decrease 193 = 0.17460001
Depth Threshold 193 = 0.0500000000
Maximum Infiltration 193 = 0.00030000
Capillary Drive 193 = 0.35835526
Surface Albedo 193 = 0.30
Number of Soil Layers 193 = 3
Porosity 193 = 0.4344 0.4344 0.4344
Pore Size Distribution 193 = 0.56246455 0.56246455 0.56246455
Bubbling Pressure 193 = 0.08922046 0.08922046 0.08922046
Field Capacity 193 = 0.1276 0.1276 0.1276
Wilting Point 193 =  0.0643 0.0643 0.0643
Bulk Density 193 = 1498.95 1498.95 1498.95
Vertical Conductivity 193 = 0.00012988 0.00012988 0.00012988
Thermal Conductivity 193 = 0.7680 0.7680 0.7680
Thermal Capacity 193 = 1128136 1128136 1128136
Mannings n 193 = 0.0300

Soil Description 194 =   74897
Lateral Conductivity 194 = 0.00000900
Exponential Decrease 194 = 0.21717209
Depth Threshold 194 = 0.0500000000
Maximum Infiltration 194 = 0.00000900
Capillary Drive 194 = 0.30282895
Surface Albedo 194 = 0.02
Number of Soil Layers 194 = 3
Porosity 194 = 0.4153 0.4153 0.4153
Pore Size Distribution 194 = 0.51020421 0.51020421 0.51020421
Bubbling Pressure 194 = 0.12322936 0.12322936 0.12322936
Field Capacity 194 = 0.1773 0.1773 0.1773
Wilting Point 194 =  0.0778 0.0778 0.0778
Bulk Density 194 = 1549.47 1549.47 1549.47
Vertical Conductivity 194 = 0.00000414 0.00000414 0.00000414
Thermal Conductivity 194 = 0.7113 0.7113 0.7113
Thermal Capacity 194 = 1354862 1354862 1354862
Mannings n 194 = 0.0300

Soil Description 195 =   74899
Lateral Conductivity 195 = 0.00000794
Exponential Decrease 195 = 1.45136217
Depth Threshold 195 = 0.0500000000
Maximum Infiltration 195 = 0.00000794
Capillary Drive 195 = 0.25293497
Surface Albedo 195 = 0.08
Number of Soil Layers 195 = 3
Porosity 195 = 0.5435 0.5435 0.5435
Pore Size Distribution 195 = 0.50945090 0.50945090 0.50945090
Bubbling Pressure 195 = 0.12371959 0.12371959 0.12371959
Field Capacity 195 = 0.1450 0.1450 0.1450
Wilting Point 195 =  0.0572 0.0572 0.0572
Bulk Density 195 = 1209.86 1209.86 1209.86
Vertical Conductivity 195 = 0.00000508 0.00000508 0.00000508
Thermal Conductivity 195 = 0.7105 0.7105 0.7105
Thermal Capacity 195 = 1358131 1358131 1358131
Mannings n 195 = 0.0300

Soil Description 196 =   74900
Lateral Conductivity 196 = 0.00002800
Exponential Decrease 196 = 0.31193545
Depth Threshold 196 = 0.0500000000
Maximum Infiltration 196 = 0.00030000
Capillary Drive 196 = 0.06723684
Surface Albedo 196 = 0.30
Number of Soil Layers 196 = 3
Porosity 196 = 0.3968 0.3968 0.3968
Pore Size Distribution 196 = 0.56806627 0.56806627 0.56806627
Bubbling Pressure 196 = 0.08557510 0.08557510 0.08557510
Field Capacity 196 = 0.0528 0.0528 0.0528
Wilting Point 196 =  0.0193 0.0193 0.0193
Bulk Density 196 = 1598.49 1598.49 1598.49
Vertical Conductivity 196 = 0.00023074 0.00023074 0.00023074
Thermal Conductivity 196 = 0.7740 0.7740 0.7740
Thermal Capacity 196 = 1103834 1103834 1103834
Mannings n 196 = 0.0300

Soil Description 197 =   74901
Lateral Conductivity 197 = 0.00002800
Exponential Decrease 197 = 0.31193545
Depth Threshold 197 = 0.0500000000
Maximum Infiltration 197 = 0.00030000
Capillary Drive 197 = 0.06723684
Surface Albedo 197 = 0.30
Number of Soil Layers 197 = 3
Porosity 197 = 0.3968 0.3968 0.3968
Pore Size Distribution 197 = 0.56806627 0.56806627 0.56806627
Bubbling Pressure 197 = 0.08557510 0.08557510 0.08557510
Field Capacity 197 = 0.0528 0.0528 0.0528
Wilting Point 197 =  0.0193 0.0193 0.0193
Bulk Density 197 = 1598.49 1598.49 1598.49
Vertical Conductivity 197 = 0.00023074 0.00023074 0.00023074
Thermal Conductivity 197 = 0.7740 0.7740 0.7740
Thermal Capacity 197 = 1103834 1103834 1103834
Mannings n 197 = 0.0300

Soil Description 198 =   74908
Lateral Conductivity 198 = 0.00009200
Exponential Decrease 198 = 0.00000000
Depth Threshold 198 = 0.0500000000
Maximum Infiltration 198 = 0.00009200
Capillary Drive 198 = 0.34480263
Surface Albedo 198 = 0.23
Number of Soil Layers 198 = 3
Porosity 198 = 0.3947 0.3947 0.3947
Pore Size Distribution 198 = 0.57964798 0.57964798 0.57964798
Bubbling Pressure 198 = 0.07803819 0.07803819 0.07803819
Field Capacity 198 = 0.0805 0.0805 0.0805
Wilting Point 198 =  0.0200 0.0200 0.0200
Bulk Density 198 = 1604.14 1604.14 1604.14
Vertical Conductivity 198 = 0.00009200 0.00009200 0.00009200
Thermal Conductivity 198 = 0.7866 0.7866 0.7866
Thermal Capacity 198 = 1053588 1053588 1053588
Mannings n 198 = 0.0300

Soil Description 199 =   74911
Lateral Conductivity 199 = 0.00000900
Exponential Decrease 199 = 0.37019740
Depth Threshold 199 = 0.0500000000
Maximum Infiltration 199 = 0.00000900
Capillary Drive 199 = 0.49815790
Surface Albedo 199 = 0.23
Number of Soil Layers 199 = 3
Porosity 199 = 0.5049 0.5049 0.5049
Pore Size Distribution 199 = 0.25031824 0.25031824 0.25031824
Bubbling Pressure 199 = 0.29235256 0.29235256 0.29235256
Field Capacity 199 = 0.2427 0.2427 0.2427
Wilting Point 199 =  0.1181 0.1181 0.1181
Bulk Density 199 = 1311.97 1311.97 1311.97
Vertical Conductivity 199 = 0.00000538 0.00000538 0.00000538
Thermal Conductivity 199 = 0.4294 0.4294 0.4294
Thermal Capacity 199 = 2482350 2482350 2482350
Mannings n 199 = 0.0300

Soil Description 200 =   74915
Lateral Conductivity 200 = 0.00002800
Exponential Decrease 200 = 0.55735436
Depth Threshold 200 = 0.0500000000
Maximum Infiltration 200 = 0.00009200
Capillary Drive 200 = 0.41552632
Surface Albedo 200 = 0.37
Number of Soil Layers 200 = 3
Porosity 200 = 0.4886 0.4886 0.4886
Pore Size Distribution 200 = 0.57936932 0.57936932 0.57936932
Bubbling Pressure 200 = 0.07821953 0.07821953 0.07821953
Field Capacity 200 = 0.0979 0.0979 0.0979
Wilting Point 200 =  0.0281 0.0281 0.0281
Bulk Density 200 = 1355.13 1355.13 1355.13
Vertical Conductivity 200 = 0.00009116 0.00009116 0.00009116
Thermal Conductivity 200 = 0.7863 0.7863 0.7863
Thermal Capacity 200 = 1054797 1054797 1054797
Mannings n 200 = 0.0300

Soil Description 201 =   74918
Lateral Conductivity 201 = 0.00000900
Exponential Decrease 201 = 0.00000000
Depth Threshold 201 = 0.0500000000
Maximum Infiltration 201 = 0.00000900
Capillary Drive 201 = 0.44361842
Surface Albedo 201 = 0.16
Number of Soil Layers 201 = 3
Porosity 201 = 0.4849 0.4849 0.4849
Pore Size Distribution 201 = 0.51194346 0.51194346 0.51194346
Bubbling Pressure 201 = 0.12209753 0.12209753 0.12209753
Field Capacity 201 = 0.1801 0.1801 0.1801
Wilting Point 201 =  0.0712 0.0712 0.0712
Bulk Density 201 = 1364.93 1364.93 1364.93
Vertical Conductivity 201 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 201 = 0.7132 0.7132 0.7132
Thermal Capacity 201 = 1347317 1347317 1347317
Mannings n 201 = 0.0300

Soil Description 202 =   74919
Lateral Conductivity 202 = 0.00000900
Exponential Decrease 202 = 3.47647318
Depth Threshold 202 = 0.0500000000
Maximum Infiltration 202 = 0.00002800
Capillary Drive 202 = 0.04342105
Surface Albedo 202 = 0.02
Number of Soil Layers 202 = 3
Porosity 202 = 0.9512 0.9512 0.9512
Pore Size Distribution 202 = 0.23965901 0.23965901 0.23965901
Bubbling Pressure 202 = 0.29928915 0.29928915 0.29928915
Field Capacity 202 = 0.1274 0.1274 0.1274
Wilting Point 202 =  0.0774 0.0774 0.0774
Bulk Density 202 = 129.28 129.28 129.28
Vertical Conductivity 202 = 0.00001088 0.00001088 0.00001088
Thermal Conductivity 202 = 0.4179 0.4179 0.4179
Thermal Capacity 202 = 2528594 2528594 2528594
Mannings n 202 = 0.0300

Soil Description 203 =   74920
Lateral Conductivity 203 = 0.00000900
Exponential Decrease 203 = 0.22108100
Depth Threshold 203 = 0.0500000000
Maximum Infiltration 203 = 0.00030000
Capillary Drive 203 = 0.35089888
Surface Albedo 203 = 0.37
Number of Soil Layers 203 = 3
Porosity 203 = 0.4843 0.4843 0.4843
Pore Size Distribution 203 = 0.56196451 0.56196451 0.56196451
Bubbling Pressure 203 = 0.08954587 0.08954587 0.08954587
Field Capacity 203 = 0.1098 0.1098 0.1098
Wilting Point 203 =  0.0401 0.0401 0.0401
Bulk Density 203 = 1366.69 1366.69 1366.69
Vertical Conductivity 203 = 0.00011425 0.00011425 0.00011425
Thermal Conductivity 203 = 0.7674 0.7674 0.7674
Thermal Capacity 203 = 1130306 1130306 1130306
Mannings n 203 = 0.0300

Soil Description 204 =   74922
Lateral Conductivity 204 = 0.00001459
Exponential Decrease 204 = 0.29396862
Depth Threshold 204 = 0.0500000000
Maximum Infiltration 204 = 0.00023882
Capillary Drive 204 = 0.36990695
Surface Albedo 204 = 0.37
Number of Soil Layers 204 = 3
Porosity 204 = 0.4856 0.4856 0.4856
Pore Size Distribution 204 = 0.56752174 0.56752174 0.56752174
Bubbling Pressure 204 = 0.08592945 0.08592945 0.08592945
Field Capacity 204 = 0.1063 0.1063 0.1063
Wilting Point 204 =  0.0365 0.0365 0.0365
Bulk Density 204 = 1363.29 1363.29 1363.29
Vertical Conductivity 204 = 0.00010746 0.00010746 0.00010746
Thermal Conductivity 204 = 0.7735 0.7735 0.7735
Thermal Capacity 204 = 1106196 1106196 1106196
Mannings n 204 = 0.0300

Soil Description 205 =   74924
Lateral Conductivity 205 = 0.00000900
Exponential Decrease 205 = 0.64712737
Depth Threshold 205 = 0.0500000000
Maximum Infiltration 205 = 0.00002800
Capillary Drive 205 = 0.40730263
Surface Albedo 205 = 0.23
Number of Soil Layers 205 = 3
Porosity 205 = 0.4729 0.4729 0.4729
Pore Size Distribution 205 = 0.54518579 0.54518579 0.54518579
Bubbling Pressure 205 = 0.10046478 0.10046478 0.10046478
Field Capacity 205 = 0.1804 0.1804 0.1804
Wilting Point 205 =  0.0802 0.0802 0.0802
Bulk Density 205 = 1396.84 1396.84 1396.84
Vertical Conductivity 205 = 0.00002488 0.00002488 0.00002488
Thermal Conductivity 205 = 0.7492 0.7492 0.7492
Thermal Capacity 205 = 1203099 1203099 1203099
Mannings n 205 = 0.0300

Soil Description 206 =   74928
Lateral Conductivity 206 = 0.00000900
Exponential Decrease 206 = 0.00000000
Depth Threshold 206 = 0.0500000000
Maximum Infiltration 206 = 0.00000900
Capillary Drive 206 = 0.25432099
Surface Albedo 206 = 0.23
Number of Soil Layers 206 = 3
Porosity 206 = 0.6171 0.6171 0.6171
Pore Size Distribution 206 = 0.50787591 0.50787591 0.50787591
Bubbling Pressure 206 = 0.12474453 0.12474453 0.12474453
Field Capacity 206 = 0.1385 0.1385 0.1385
Wilting Point 206 =  0.0388 0.0388 0.0388
Bulk Density 206 = 1014.57 1014.57 1014.57
Vertical Conductivity 206 = 0.00000900 0.00000900 0.00000900
Thermal Conductivity 206 = 0.7088 0.7088 0.7088
Thermal Capacity 206 = 1364964 1364964 1364964
Mannings n 206 = 0.0300

Soil Description 207 =   74929
Lateral Conductivity 207 = 0.00000800
Exponential Decrease 207 = 0.50000000
Depth Threshold 207 = 0.0500000000
Maximum Infiltration 207 = 0.00000800
Capillary Drive 207 = 0.23953753
Surface Albedo 207 = 0.15
Number of Soil Layers 207 = 3
Porosity 207 = 0.6036 0.6036 0.6036
Pore Size Distribution 207 = 0.50862589 0.50862589 0.50862589
Bubbling Pressure 207 = 0.12425647 0.12425647 0.12425647
Field Capacity 207 = 0.1339 0.1339 0.1339
Wilting Point 207 =  0.0453 0.0453 0.0453
Bulk Density 207 = 1050.42 1050.42 1050.42
Vertical Conductivity 207 = 0.00000665 0.00000665 0.00000665
Thermal Conductivity 207 = 0.7096 0.7096 0.7096
Thermal Capacity 207 = 1361710 1361710 1361710
Mannings n 207 = 0.0300

Soil Description 208 =   74933
Lateral Conductivity 208 = 0.00000900
Exponential Decrease 208 = 0.39324466
Depth Threshold 208 = 0.0500000000
Maximum Infiltration 208 = 0.00000900
Capillary Drive 208 = 0.50993421
Surface Albedo 208 = 0.23
Number of Soil Layers 208 = 3
Porosity 208 = 0.5067 0.5067 0.5067
Pore Size Distribution 208 = 0.33781241 0.33781241 0.33781241
Bubbling Pressure 208 = 0.23541492 0.23541492 0.23541492
Field Capacity 208 = 0.2676 0.2676 0.2676
Wilting Point 208 =  0.1147 0.1147 0.1147
Bulk Density 208 = 1307.37 1307.37 1307.37
Vertical Conductivity 208 = 0.00000485 0.00000485 0.00000485
Thermal Conductivity 208 = 0.5243 0.5243 0.5243
Thermal Capacity 208 = 2102766 2102766 2102766
Mannings n 208 = 0.0300

Soil Description 209 =   74934
Lateral Conductivity 209 = 0.00000900
Exponential Decrease 209 = 0.39324466
Depth Threshold 209 = 0.0500000000
Maximum Infiltration 209 = 0.00000900
Capillary Drive 209 = 0.50993421
Surface Albedo 209 = 0.23
Number of Soil Layers 209 = 3
Porosity 209 = 0.5067 0.5067 0.5067
Pore Size Distribution 209 = 0.33781241 0.33781241 0.33781241
Bubbling Pressure 209 = 0.23541492 0.23541492 0.23541492
Field Capacity 209 = 0.2676 0.2676 0.2676
Wilting Point 209 =  0.1147 0.1147 0.1147
Bulk Density 209 = 1307.37 1307.37 1307.37
Vertical Conductivity 209 = 0.00000485 0.00000485 0.00000485
Thermal Conductivity 209 = 0.5243 0.5243 0.5243
Thermal Capacity 209 = 2102766 2102766 2102766
Mannings n 209 = 0.0300

Soil Description 210 =   74935
Lateral Conductivity 210 = 0.00000900
Exponential Decrease 210 = 0.39324466
Depth Threshold 210 = 0.0500000000
Maximum Infiltration 210 = 0.00000900
Capillary Drive 210 = 0.50993421
Surface Albedo 210 = 0.23
Number of Soil Layers 210 = 3
Porosity 210 = 0.5067 0.5067 0.5067
Pore Size Distribution 210 = 0.33781241 0.33781241 0.33781241
Bubbling Pressure 210 = 0.23541492 0.23541492 0.23541492
Field Capacity 210 = 0.2676 0.2676 0.2676
Wilting Point 210 =  0.1147 0.1147 0.1147
Bulk Density 210 = 1307.37 1307.37 1307.37
Vertical Conductivity 210 = 0.00000485 0.00000485 0.00000485
Thermal Conductivity 210 = 0.5243 0.5243 0.5243
Thermal Capacity 210 = 2102766 2102766 2102766
Mannings n 210 = 0.0300

Soil Description 211 =   74937
Lateral Conductivity 211 = 0.00009200
Exponential Decrease 211 = 0.58087346
Depth Threshold 211 = 0.0500000000
Maximum Infiltration 211 = 0.00030000
Capillary Drive 211 = 0.30684210
Surface Albedo 211 = 0.30
Number of Soil Layers 211 = 3
Porosity 211 = 0.3768 0.3768 0.3768
Pore Size Distribution 211 = 0.57933636 0.57933636 0.57933636
Bubbling Pressure 211 = 0.07824098 0.07824098 0.07824098
Field Capacity 211 = 0.0937 0.0937 0.0937
Wilting Point 211 =  0.0273 0.0273 0.0273
Bulk Density 211 = 1651.38 1651.38 1651.38
Vertical Conductivity 211 = 0.00012621 0.00012621 0.00012621
Thermal Conductivity 211 = 0.7863 0.7863 0.7863
Thermal Capacity 211 = 1054940 1054940 1054940
Mannings n 211 = 0.0300

Soil Description 212 =   74943
Lateral Conductivity 212 = 0.00002800
Exponential Decrease 212 = 0.86322026
Depth Threshold 212 = 0.0500000000
Maximum Infiltration 212 = 0.00009200
Capillary Drive 212 = 0.41671053
Surface Albedo 212 = 0.23
Number of Soil Layers 212 = 3
Porosity 212 = 0.4604 0.4604 0.4604
Pore Size Distribution 212 = 0.55849709 0.55849709 0.55849709
Bubbling Pressure 212 = 0.09180233 0.09180233 0.09180233
Field Capacity 212 = 0.1302 0.1302 0.1302
Wilting Point 212 =  0.0549 0.0549 0.0549
Bulk Density 212 = 1430.00 1430.00 1430.00
Vertical Conductivity 212 = 0.00006000 0.00006000 0.00006000
Thermal Conductivity 212 = 0.7637 0.7637 0.7637
Thermal Capacity 212 = 1145349 1145349 1145349
Mannings n 212 = 0.0300

Soil Description 213 =   74944
Lateral Conductivity 213 = 0.00002800
Exponential Decrease 213 = 0.76302111
Depth Threshold 213 = 0.0500000000
Maximum Infiltration 213 = 0.00009200
Capillary Drive 213 = 0.41881579
Surface Albedo 213 = 0.23
Number of Soil Layers 213 = 3
Porosity 213 = 0.4841 0.4841 0.4841
Pore Size Distribution 213 = 0.56052043 0.56052043 0.56052043
Bubbling Pressure 213 = 0.09048562 0.09048562 0.09048562
Field Capacity 213 = 0.1210 0.1210 0.1210
Wilting Point 213 =  0.0389 0.0389 0.0389
Bulk Density 213 = 1367.11 1367.11 1367.11
Vertical Conductivity 213 = 0.00006632 0.00006632 0.00006632
Thermal Conductivity 213 = 0.7659 0.7659 0.7659
Thermal Capacity 213 = 1136571 1136571 1136571
Mannings n 213 = 0.0300

Soil Description 214 =   74953
Lateral Conductivity 214 = 0.00000900
Exponential Decrease 214 = 0.66777403
Depth Threshold 214 = 0.0500000000
Maximum Infiltration 214 = 0.00002800
Capillary Drive 214 = 0.35578947
Surface Albedo 214 = 0.23
Number of Soil Layers 214 = 3
Porosity 214 = 0.5978 0.5978 0.5978
Pore Size Distribution 214 = 0.53030028 0.53030028 0.53030028
Bubbling Pressure 214 = 0.11015166 0.11015166 0.11015166
Field Capacity 214 = 0.1203 0.1203 0.1203
Wilting Point 214 =  0.0421 0.0421 0.0421
Bulk Density 214 = 1065.79 1065.79 1065.79
Vertical Conductivity 214 = 0.00002411 0.00002411 0.00002411
Thermal Conductivity 214 = 0.7331 0.7331 0.7331
Thermal Capacity 214 = 1267678 1267678 1267678
Mannings n 214 = 0.0300

Soil Description 215 =   74954
Lateral Conductivity 215 = 0.00000900
Exponential Decrease 215 = 0.95826683
Depth Threshold 215 = 0.0500000000
Maximum Infiltration 215 = 0.00000900
Capillary Drive 215 = 0.48927632
Surface Albedo 215 = 0.30
Number of Soil Layers 215 = 3
Porosity 215 = 0.4902 0.4902 0.4902
Pore Size Distribution 215 = 0.35990006 0.35990006 0.35990006
Bubbling Pressure 215 = 0.22104118 0.22104118 0.22104118
Field Capacity 215 = 0.2806 0.2806 0.2806
Wilting Point 215 =  0.1331 0.1331 0.1331
Bulk Density 215 = 1350.99 1350.99 1350.99
Vertical Conductivity 215 = 0.00000596 0.00000596 0.00000596
Thermal Conductivity 215 = 0.5483 0.5483 0.5483
Thermal Capacity 215 = 2006941 2006941 2006941
Mannings n 215 = 0.0300

Soil Description 216 =   74955
Lateral Conductivity 216 = 0.00000900
Exponential Decrease 216 = 0.28400100
Depth Threshold 216 = 0.0500000000
Maximum Infiltration 216 = 0.00030000
Capillary Drive 216 = 0.36394737
Surface Albedo 216 = 0.23
Number of Soil Layers 216 = 3
Porosity 216 = 0.4316 0.4316 0.4316
Pore Size Distribution 216 = 0.55901527 0.55901527 0.55901527
Bubbling Pressure 216 = 0.09146512 0.09146512 0.09146512
Field Capacity 216 = 0.1298 0.1298 0.1298
Wilting Point 216 =  0.0452 0.0452 0.0452
Bulk Density 216 = 1506.32 1506.32 1506.32
Vertical Conductivity 216 = 0.00009126 0.00009126 0.00009126
Thermal Conductivity 216 = 0.7642 0.7642 0.7642
Thermal Capacity 216 = 1143101 1143101 1143101
Mannings n 216 = 0.0300

Soil Description 217 =   74961
Lateral Conductivity 217 = 0.00036160
Exponential Decrease 217 = 0.17500818
Depth Threshold 217 = 0.0500000000
Maximum Infiltration 217 = 0.00036474
Capillary Drive 217 = 0.32771411
Surface Albedo 217 = 0.22
Number of Soil Layers 217 = 3
Porosity 217 = 0.5055 0.5055 0.5055
Pore Size Distribution 217 = 0.55024757 0.55024757 0.55024757
Bubbling Pressure 217 = 0.09717078 0.09717078 0.09717078
Field Capacity 217 = 0.0908 0.0908 0.0908
Wilting Point 217 =  0.0344 0.0344 0.0344
Bulk Density 217 = 1310.38 1310.38 1310.38
Vertical Conductivity 217 = 0.00004029 0.00004029 0.00004029
Thermal Conductivity 217 = 0.7547 0.7547 0.7547
Thermal Capacity 217 = 1181139 1181139 1181139
Mannings n 217 = 0.0300

Soil Description 218 =   74962
Lateral Conductivity 218 = 0.00036160
Exponential Decrease 218 = 0.19833682
Depth Threshold 218 = 0.0500000000
Maximum Infiltration 218 = 0.00036474
Capillary Drive 218 = 0.31643199
Surface Albedo 218 = 0.23
Number of Soil Layers 218 = 3
Porosity 218 = 0.5081 0.5081 0.5081
Pore Size Distribution 218 = 0.54273446 0.54273446 0.54273446
Bubbling Pressure 218 = 0.10206000 0.10206000 0.10206000
Field Capacity 218 = 0.0996 0.0996 0.0996
Wilting Point 218 =  0.0376 0.0376 0.0376
Bulk Density 218 = 1303.42 1303.42 1303.42
Vertical Conductivity 218 = 0.00004061 0.00004061 0.00004061
Thermal Conductivity 218 = 0.7466 0.7466 0.7466
Thermal Capacity 218 = 1213733 1213733 1213733
Mannings n 218 = 0.0300

Soil Description 219 =   74965
Lateral Conductivity 219 = 0.00000794
Exponential Decrease 219 = 1.25070219
Depth Threshold 219 = 0.0500000000
Maximum Infiltration 219 = 0.00000794
Capillary Drive 219 = 0.18619872
Surface Albedo 219 = 0.20
Number of Soil Layers 219 = 3
Porosity 219 = 0.5540 0.5540 0.5540
Pore Size Distribution 219 = 0.52518540 0.52518540 0.52518540
Bubbling Pressure 219 = 0.11348021 0.11348021 0.11348021
Field Capacity 219 = 0.1170 0.1170 0.1170
Wilting Point 219 =  0.0449 0.0449 0.0449
Bulk Density 219 = 1181.84 1181.84 1181.84
Vertical Conductivity 219 = 0.00000546 0.00000546 0.00000546
Thermal Conductivity 219 = 0.7275 0.7275 0.7275
Thermal Capacity 219 = 1289868 1289868 1289868
Mannings n 219 = 0.0300

Soil Description 220 =   74966
Lateral Conductivity 220 = 0.00000900
Exponential Decrease 220 = 0.21999405
Depth Threshold 220 = 0.0500000000
Maximum Infiltration 220 = 0.00010600
Capillary Drive 220 = 0.22390351
Surface Albedo 220 = 0.21
Number of Soil Layers 220 = 3
Porosity 220 = 0.4306 0.4306 0.4306
Pore Size Distribution 220 = 0.54442001 0.54442001 0.54442001
Bubbling Pressure 220 = 0.10096312 0.10096312 0.10096312
Field Capacity 220 = 0.1231 0.1231 0.1231
Wilting Point 220 =  0.0485 0.0485 0.0485
Bulk Density 220 = 1508.90 1508.90 1508.90
Vertical Conductivity 220 = 0.00006234 0.00006234 0.00006234
Thermal Conductivity 220 = 0.7484 0.7484 0.7484
Thermal Capacity 220 = 1206421 1206421 1206421
Mannings n 220 = 0.0300

Soil Description 221 =   74970
Lateral Conductivity 221 = 0.00000900
Exponential Decrease 221 = 0.22065174
Depth Threshold 221 = 0.0500000000
Maximum Infiltration 221 = 0.00030000
Capillary Drive 221 = 0.15894737
Surface Albedo 221 = 0.16
Number of Soil Layers 221 = 3
Porosity 221 = 0.4107 0.4107 0.4107
Pore Size Distribution 221 = 0.56184784 0.56184784 0.56184784
Bubbling Pressure 221 = 0.08962180 0.08962180 0.08962180
Field Capacity 221 = 0.0893 0.0893 0.0893
Wilting Point 221 =  0.0257 0.0257 0.0257
Bulk Density 221 = 1561.58 1561.58 1561.58
Vertical Conductivity 221 = 0.00017747 0.00017747 0.00017747
Thermal Conductivity 221 = 0.7673 0.7673 0.7673
Thermal Capacity 221 = 1130812 1130812 1130812
Mannings n 221 = 0.0300

Soil Description 222 =   74971
Lateral Conductivity 222 = 0.00000900
Exponential Decrease 222 = 0.22065174
Depth Threshold 222 = 0.0500000000
Maximum Infiltration 222 = 0.00030000
Capillary Drive 222 = 0.15894737
Surface Albedo 222 = 0.16
Number of Soil Layers 222 = 3
Porosity 222 = 0.4107 0.4107 0.4107
Pore Size Distribution 222 = 0.56184784 0.56184784 0.56184784
Bubbling Pressure 222 = 0.08962180 0.08962180 0.08962180
Field Capacity 222 = 0.0893 0.0893 0.0893
Wilting Point 222 =  0.0257 0.0257 0.0257
Bulk Density 222 = 1561.58 1561.58 1561.58
Vertical Conductivity 222 = 0.00017747 0.00017747 0.00017747
Thermal Conductivity 222 = 0.7673 0.7673 0.7673
Thermal Capacity 222 = 1130812 1130812 1130812
Mannings n 222 = 0.0300

Soil Description 223 =   74972
Lateral Conductivity 223 = 0.00000900
Exponential Decrease 223 = 0.50000000
Depth Threshold 223 = 0.0500000000
Maximum Infiltration 223 = 0.00000900
Capillary Drive 223 = 0.46296053
Surface Albedo 223 = 0.16
Number of Soil Layers 223 = 3
Porosity 223 = 0.4463 0.4463 0.4463
Pore Size Distribution 223 = 0.30285479 0.30285479 0.30285479
Bubbling Pressure 223 = 0.25816391 0.25816391 0.25816391
Field Capacity 223 = 0.3112 0.3112 0.3112
Wilting Point 223 =  0.1740 0.1740 0.1740
Bulk Density 223 = 1467.30 1467.30 1467.30
Vertical Conductivity 223 = 0.00000810 0.00000810 0.00000810
Thermal Conductivity 223 = 0.4864 0.4864 0.4864
Thermal Capacity 223 = 2254426 2254426 2254426
Mannings n 223 = 0.0300

Soil Description 224 =   75961
Lateral Conductivity 224 = 0.00002505
Exponential Decrease 224 = 0.00000000
Depth Threshold 224 = 0.0500000000
Maximum Infiltration 224 = 0.00002505
Capillary Drive 224 = 0.26139907
Surface Albedo 224 = 0.08
Number of Soil Layers 224 = 3
Porosity 224 = 0.5289 0.5289 0.5289
Pore Size Distribution 224 = 0.53205202 0.53205202 0.53205202
Bubbling Pressure 224 = 0.10901170 0.10901170 0.10901170
Field Capacity 224 = 0.0795 0.0795 0.0795
Wilting Point 224 =  0.0319 0.0319 0.0319
Bulk Density 224 = 1248.44 1248.44 1248.44
Vertical Conductivity 224 = 0.00002505 0.00002505 0.00002505
Thermal Conductivity 224 = 0.7350 0.7350 0.7350
Thermal Capacity 224 = 1260078 1260078 1260078
Mannings n 224 = 0.0300

Soil Description 225 =   75962
Lateral Conductivity 225 = 0.00002489
Exponential Decrease 225 = 0.00000000
Depth Threshold 225 = 0.0500000000
Maximum Infiltration 225 = 0.00002489
Capillary Drive 225 = 0.32218537
Surface Albedo 225 = 0.31
Number of Soil Layers 225 = 3
Porosity 225 = 0.5789 0.5789 0.5789
Pore Size Distribution 225 = 0.55339287 0.55339287 0.55339287
Bubbling Pressure 225 = 0.09512395 0.09512395 0.09512395
Field Capacity 225 = 0.0751 0.0751 0.0751
Wilting Point 225 =  0.0238 0.0238 0.0238
Bulk Density 225 = 1115.86 1115.86 1115.86
Vertical Conductivity 225 = 0.00002489 0.00002489 0.00002489
Thermal Conductivity 225 = 0.7581 0.7581 0.7581
Thermal Capacity 225 = 1167493 1167493 1167493
Mannings n 225 = 0.0300

Soil Description 226 =   75963
Lateral Conductivity 226 = 0.00002705
Exponential Decrease 226 = 0.50000000
Depth Threshold 226 = 0.0500000000
Maximum Infiltration 226 = 0.00003523
Capillary Drive 226 = 0.31670967
Surface Albedo 226 = 0.34
Number of Soil Layers 226 = 3
Porosity 226 = 0.5656 0.5656 0.5656
Pore Size Distribution 226 = 0.55180673 0.55180673 0.55180673
Bubbling Pressure 226 = 0.09615614 0.09615614 0.09615614
Field Capacity 226 = 0.0781 0.0781 0.0781
Wilting Point 226 =  0.0268 0.0268 0.0268
Bulk Density 226 = 1151.21 1151.21 1151.21
Vertical Conductivity 226 = 0.00003141 0.00003141 0.00003141
Thermal Conductivity 226 = 0.7564 0.7564 0.7564
Thermal Capacity 226 = 1174374 1174374 1174374
Mannings n 226 = 0.0300

Soil Description 227 =   75964
Lateral Conductivity 227 = 0.00002283
Exponential Decrease 227 = 0.00000000
Depth Threshold 227 = 0.0500000000
Maximum Infiltration 227 = 0.00002283
Capillary Drive 227 = 0.21814205
Surface Albedo 227 = 0.27
Number of Soil Layers 227 = 3
Porosity 227 = 0.5467 0.5467 0.5467
Pore Size Distribution 227 = 0.54800578 0.54800578 0.54800578
Bubbling Pressure 227 = 0.09862965 0.09862965 0.09862965
Field Capacity 227 = 0.0569 0.0569 0.0569
Wilting Point 227 =  0.0190 0.0190 0.0190
Bulk Density 227 = 1201.36 1201.36 1201.36
Vertical Conductivity 227 = 0.00002283 0.00002283 0.00002283
Thermal Conductivity 227 = 0.7523 0.7523 0.7523
Thermal Capacity 227 = 1190864 1190864 1190864
Mannings n 227 = 0.0300

Soil Description 228 =   75965
Lateral Conductivity 228 = 0.00002306
Exponential Decrease 228 = 1.20386444
Depth Threshold 228 = 0.0500000000
Maximum Infiltration 228 = 0.00005694
Capillary Drive 228 = 0.25458949
Surface Albedo 228 = 0.22
Number of Soil Layers 228 = 3
Porosity 228 = 0.5605 0.5605 0.5605
Pore Size Distribution 228 = 0.54805070 0.54805070 0.54805070
Bubbling Pressure 228 = 0.09860041 0.09860041 0.09860041
Field Capacity 228 = 0.0726 0.0726 0.0726
Wilting Point 228 =  0.0247 0.0247 0.0247
Bulk Density 228 = 1164.65 1164.65 1164.65
Vertical Conductivity 228 = 0.00005137 0.00005137 0.00005137
Thermal Conductivity 228 = 0.7523 0.7523 0.7523
Thermal Capacity 228 = 1190669 1190669 1190669
Mannings n 228 = 0.0300

Soil Description 229 =   75966
Lateral Conductivity 229 = 0.00002559
Exponential Decrease 229 = 0.00000000
Depth Threshold 229 = 0.0500000000
Maximum Infiltration 229 = 0.00002559
Capillary Drive 229 = 0.25114671
Surface Albedo 229 = 0.31
Number of Soil Layers 229 = 3
Porosity 229 = 0.4927 0.4927 0.4927
Pore Size Distribution 229 = 0.54829678 0.54829678 0.54829678
Bubbling Pressure 229 = 0.09844027 0.09844027 0.09844027
Field Capacity 229 = 0.0658 0.0658 0.0658
Wilting Point 229 =  0.0221 0.0221 0.0221
Bulk Density 229 = 1344.32 1344.32 1344.32
Vertical Conductivity 229 = 0.00002559 0.00002559 0.00002559
Thermal Conductivity 229 = 0.7526 0.7526 0.7526
Thermal Capacity 229 = 1189602 1189602 1189602
Mannings n 229 = 0.0300

Soil Description 230 =   75967
Lateral Conductivity 230 = 0.00002587
Exponential Decrease 230 = 0.00000000
Depth Threshold 230 = 0.0500000000
Maximum Infiltration 230 = 0.00002587
Capillary Drive 230 = 0.25803213
Surface Albedo 230 = 0.32
Number of Soil Layers 230 = 3
Porosity 230 = 0.4878 0.4878 0.4878
Pore Size Distribution 230 = 0.54847910 0.54847910 0.54847910
Bubbling Pressure 230 = 0.09832162 0.09832162 0.09832162
Field Capacity 230 = 0.0678 0.0678 0.0678
Wilting Point 230 =  0.0228 0.0228 0.0228
Bulk Density 230 = 1357.31 1357.31 1357.31
Vertical Conductivity 230 = 0.00002587 0.00002587 0.00002587
Thermal Conductivity 230 = 0.7528 0.7528 0.7528
Thermal Capacity 230 = 1188811 1188811 1188811
Mannings n 230 = 0.0300

Soil Description 231 =   75968
Lateral Conductivity 231 = 0.00001153
Exponential Decrease 231 = 0.00000000
Depth Threshold 231 = 0.0500000000
Maximum Infiltration 231 = 0.00001153
Capillary Drive 231 = 0.12561807
Surface Albedo 231 = 0.09
Number of Soil Layers 231 = 3
Porosity 231 = 0.7872 0.7872 0.7872
Pore Size Distribution 231 = 0.54980650 0.54980650 0.54980650
Bubbling Pressure 231 = 0.09745781 0.09745781 0.09745781
Field Capacity 231 = 0.0358 0.0358 0.0358
Wilting Point 231 =  0.0123 0.0123 0.0123
Bulk Density 231 = 563.94 563.94 563.94
Vertical Conductivity 231 = 0.00001153 0.00001153 0.00001153
Thermal Conductivity 231 = 0.7542 0.7542 0.7542
Thermal Capacity 231 = 1183052 1183052 1183052
Mannings n 231 = 0.0300

Soil Description 232 =   75972
Lateral Conductivity 232 = 0.00001292
Exponential Decrease 232 = 1.33458591
Depth Threshold 232 = 0.0500000000
Maximum Infiltration 232 = 0.00004246
Capillary Drive 232 = 0.14963563
Surface Albedo 232 = 0.17
Number of Soil Layers 232 = 3
Porosity 232 = 0.7520 0.7520 0.7520
Pore Size Distribution 232 = 0.55675901 0.55675901 0.55675901
Bubbling Pressure 232 = 0.09293340 0.09293340 0.09293340
Field Capacity 232 = 0.0412 0.0412 0.0412
Wilting Point 232 =  0.0127 0.0127 0.0127
Bulk Density 232 = 657.24 657.24 657.24
Vertical Conductivity 232 = 0.00003760 0.00003760 0.00003760
Thermal Conductivity 232 = 0.7618 0.7618 0.7618
Thermal Capacity 232 = 1152889 1152889 1152889
Mannings n 232 = 0.0300

Soil Description 233 =   75975
Lateral Conductivity 233 = 0.00002800
Exponential Decrease 233 = 0.11530702
Depth Threshold 233 = 0.0500000000
Maximum Infiltration 233 = 0.00002800
Capillary Drive 233 = 0.29865790
Surface Albedo 233 = 0.28
Number of Soil Layers 233 = 3
Porosity 233 = 0.4586 0.4586 0.4586
Pore Size Distribution 233 = 0.55925628 0.55925628 0.55925628
Bubbling Pressure 233 = 0.09130828 0.09130828 0.09130828
Field Capacity 233 = 0.0725 0.0725 0.0725
Wilting Point 233 =  0.0217 0.0217 0.0217
Bulk Density 233 = 1434.61 1434.61 1434.61
Vertical Conductivity 233 = 0.00001503 0.00001503 0.00001503
Thermal Conductivity 233 = 0.7645 0.7645 0.7645
Thermal Capacity 233 = 1142055 1142055 1142055
Mannings n 233 = 0.0300

Soil Description 234 =   75976
Lateral Conductivity 234 = 0.00002505
Exponential Decrease 234 = 0.00000000
Depth Threshold 234 = 0.0500000000
Maximum Infiltration 234 = 0.00002505
Capillary Drive 234 = 0.30941760
Surface Albedo 234 = 0.39
Number of Soil Layers 234 = 3
Porosity 234 = 0.5656 0.5656 0.5656
Pore Size Distribution 234 = 0.54934922 0.54934922 0.54934922
Bubbling Pressure 234 = 0.09775539 0.09775539 0.09775539
Field Capacity 234 = 0.0760 0.0760 0.0760
Wilting Point 234 =  0.0246 0.0246 0.0246
Bulk Density 234 = 1151.28 1151.28 1151.28
Vertical Conductivity 234 = 0.00002505 0.00002505 0.00002505
Thermal Conductivity 234 = 0.7537 0.7537 0.7537
Thermal Capacity 234 = 1185036 1185036 1185036
Mannings n 234 = 0.0300

Soil Description 235 =   75981
Lateral Conductivity 235 = 0.00002100
Exponential Decrease 235 = 0.00000000
Depth Threshold 235 = 0.0500000000
Maximum Infiltration 235 = 0.00002100
Capillary Drive 235 = 0.26091331
Surface Albedo 235 = 0.17
Number of Soil Layers 235 = 3
Porosity 235 = 0.6379 0.6379 0.6379
Pore Size Distribution 235 = 0.54741217 0.54741217 0.54741217
Bubbling Pressure 235 = 0.09901594 0.09901594 0.09901594
Field Capacity 235 = 0.0646 0.0646 0.0646
Wilting Point 235 =  0.0222 0.0222 0.0222
Bulk Density 235 = 959.55 959.55 959.55
Vertical Conductivity 235 = 0.00002100 0.00002100 0.00002100
Thermal Conductivity 235 = 0.7516 0.7516 0.7516
Thermal Capacity 235 = 1193440 1193440 1193440
Mannings n 235 = 0.0300

Soil Description 236 =   75982
Lateral Conductivity 236 = 0.00002489
Exponential Decrease 236 = 0.00000000
Depth Threshold 236 = 0.0500000000
Maximum Infiltration 236 = 0.00002489
Capillary Drive 236 = 0.31106000
Surface Albedo 236 = 0.21
Number of Soil Layers 236 = 3
Porosity 236 = 0.5721 0.5721 0.5721
Pore Size Distribution 236 = 0.54791148 0.54791148 0.54791148
Bubbling Pressure 236 = 0.09869101 0.09869101 0.09869101
Field Capacity 236 = 0.0766 0.0766 0.0766
Wilting Point 236 =  0.0262 0.0262 0.0262
Bulk Density 236 = 1133.99 1133.99 1133.99
Vertical Conductivity 236 = 0.00002489 0.00002489 0.00002489
Thermal Conductivity 236 = 0.7522 0.7522 0.7522
Thermal Capacity 236 = 1191273 1191273 1191273
Mannings n 236 = 0.0300

Soil Description 237 =   75983
Lateral Conductivity 237 = 0.00002068
Exponential Decrease 237 = 0.00000000
Depth Threshold 237 = 0.0500000000
Maximum Infiltration 237 = 0.00002068
Capillary Drive 237 = 0.28128303
Surface Albedo 237 = 0.22
Number of Soil Layers 237 = 3
Porosity 237 = 0.6597 0.6597 0.6597
Pore Size Distribution 237 = 0.55554987 0.55554987 0.55554987
Bubbling Pressure 237 = 0.09372026 0.09372026 0.09372026
Field Capacity 237 = 0.0636 0.0636 0.0636
Wilting Point 237 =  0.0200 0.0200 0.0200
Bulk Density 237 = 901.74 901.74 901.74
Vertical Conductivity 237 = 0.00002068 0.00002068 0.00002068
Thermal Conductivity 237 = 0.7605 0.7605 0.7605
Thermal Capacity 237 = 1158135 1158135 1158135
Mannings n 237 = 0.0300

Soil Description 238 =   75984
Lateral Conductivity 238 = 0.00001957
Exponential Decrease 238 = 0.00000000
Depth Threshold 238 = 0.0500000000
Maximum Infiltration 238 = 0.00001957
Capillary Drive 238 = 0.26616029
Surface Albedo 238 = 0.21
Number of Soil Layers 238 = 3
Porosity 238 = 0.6780 0.6780 0.6780
Pore Size Distribution 238 = 0.55554987 0.55554987 0.55554987
Bubbling Pressure 238 = 0.09372026 0.09372026 0.09372026
Field Capacity 238 = 0.0602 0.0602 0.0602
Wilting Point 238 =  0.0189 0.0189 0.0189
Bulk Density 238 = 853.26 853.26 853.26
Vertical Conductivity 238 = 0.00001957 0.00001957 0.00001957
Thermal Conductivity 238 = 0.7605 0.7605 0.7605
Thermal Capacity 238 = 1158135 1158135 1158135
Mannings n 238 = 0.0300

Soil Description 239 =   75990
Lateral Conductivity 239 = 0.00001179
Exponential Decrease 239 = 0.00000000
Depth Threshold 239 = 0.0500000000
Maximum Infiltration 239 = 0.00001179
Capillary Drive 239 = 0.15505721
Surface Albedo 239 = 0.13
Number of Soil Layers 239 = 3
Porosity 239 = 0.8542 0.8542 0.8542
Pore Size Distribution 239 = 0.56584397 0.56584397 0.56584397
Bubbling Pressure 239 = 0.08702128 0.08702128 0.08702128
Field Capacity 239 = 0.0273 0.0273 0.0273
Wilting Point 239 =  0.0072 0.0072 0.0072
Bulk Density 239 = 386.45 386.45 386.45
Vertical Conductivity 239 = 0.00001179 0.00001179 0.00001179
Thermal Conductivity 239 = 0.7716 0.7716 0.7716
Thermal Capacity 239 = 1113475 1113475 1113475
Mannings n 239 = 0.0300

Soil Description 240 =   75991
Lateral Conductivity 240 = 0.00001032
Exponential Decrease 240 = 0.00000000
Depth Threshold 240 = 0.0500000000
Maximum Infiltration 240 = 0.00001032
Capillary Drive 240 = 0.10390406
Surface Albedo 240 = 0.03
Number of Soil Layers 240 = 3
Porosity 240 = 0.8058 0.8058 0.8058
Pore Size Distribution 240 = 0.53205202 0.53205202 0.53205202
Bubbling Pressure 240 = 0.10901170 0.10901170 0.10901170
Field Capacity 240 = 0.0318 0.0318 0.0318
Wilting Point 240 =  0.0128 0.0128 0.0128
Bulk Density 240 = 514.53 514.53 514.53
Vertical Conductivity 240 = 0.00001032 0.00001032 0.00001032
Thermal Conductivity 240 = 0.7350 0.7350 0.7350
Thermal Capacity 240 = 1260078 1260078 1260078
Mannings n 240 = 0.0300

Soil Description 241 =   75992
Lateral Conductivity 241 = 0.00002465
Exponential Decrease 241 = 0.25832909
Depth Threshold 241 = 0.0500000000
Maximum Infiltration 241 = 0.00006188
Capillary Drive 241 = 0.30276703
Surface Albedo 241 = 0.32
Number of Soil Layers 241 = 3
Porosity 241 = 0.4407 0.4407 0.4407
Pore Size Distribution 241 = 0.55697448 0.55697448 0.55697448
Bubbling Pressure 241 = 0.09279318 0.09279318 0.09279318
Field Capacity 241 = 0.0853 0.0853 0.0853
Wilting Point 241 =  0.0268 0.0268 0.0268
Bulk Density 241 = 1482.16 1482.16 1482.16
Vertical Conductivity 241 = 0.00002080 0.00002080 0.00002080
Thermal Conductivity 241 = 0.7620 0.7620 0.7620
Thermal Capacity 241 = 1151955 1151955 1151955
Mannings n 241 = 0.0300

Soil Description 242 =   75993
Lateral Conductivity 242 = 0.00001768
Exponential Decrease 242 = 0.00000000
Depth Threshold 242 = 0.0500000000
Maximum Infiltration 242 = 0.00001768
Capillary Drive 242 = 0.21841242
Surface Albedo 242 = 0.28
Number of Soil Layers 242 = 3
Porosity 242 = 0.6933 0.6933 0.6933
Pore Size Distribution 242 = 0.54934922 0.54934922 0.54934922
Bubbling Pressure 242 = 0.09775539 0.09775539 0.09775539
Field Capacity 242 = 0.0537 0.0537 0.0537
Wilting Point 242 =  0.0173 0.0173 0.0173
Bulk Density 242 = 812.67 812.67 812.67
Vertical Conductivity 242 = 0.00001768 0.00001768 0.00001768
Thermal Conductivity 242 = 0.7537 0.7537 0.7537
Thermal Capacity 242 = 1185036 1185036 1185036
Mannings n 242 = 0.0300

Soil Description 243 =   75994
Lateral Conductivity 243 = 0.00002800
Exponential Decrease 243 = 0.11102880
Depth Threshold 243 = 0.0500000000
Maximum Infiltration 243 = 0.00006941
Capillary Drive 243 = 0.34229876
Surface Albedo 243 = 0.39
Number of Soil Layers 243 = 3
Porosity 243 = 0.4962 0.4962 0.4962
Pore Size Distribution 243 = 0.56065013 0.56065013 0.56065013
Bubbling Pressure 243 = 0.09040122 0.09040122 0.09040122
Field Capacity 243 = 0.0844 0.0844 0.0844
Wilting Point 243 =  0.0246 0.0246 0.0246
Bulk Density 243 = 1335.13 1335.13 1335.13
Vertical Conductivity 243 = 0.00001872 0.00001872 0.00001872
Thermal Conductivity 243 = 0.7660 0.7660 0.7660
Thermal Capacity 243 = 1136008 1136008 1136008
Mannings n 243 = 0.0300

Soil Description 244 =    9999
Lateral Conductivity 244 = 0.00023289
Exponential Decrease 244 = 1.40240405
Depth Threshold 244 = 0.0500000000
Maximum Infiltration 244 = 0.00024752
Capillary Drive 244 = 0.28740196
Surface Albedo 244 = 0.21
Number of Soil Layers 244 = 3
Porosity 244 = 0.6318 0.6318 0.6318
Pore Size Distribution 244 = 0.57232806 0.57232806 0.57232806
Bubbling Pressure 244 = 0.08280169 0.08280169 0.08280169
Field Capacity 244 = 0.0941 0.0941 0.0941
Wilting Point 244 =  0.0367 0.0367 0.0367
Bulk Density 244 = 975.81 975.81 975.81
Vertical Conductivity 244 = 0.00005390 0.00005390 0.00005390
Thermal Conductivity 244 = 0.7787 0.7787 0.7787
Thermal Capacity 244 = 1085345 1085345 1085345
Mannings n 244 = 0.0364


################################################################################
# VEGETATION INFORMATION SECTION
################################################################################
[VEGETATION]

Vegetation Map File        = arc/skagit150_nlcd2011.bin

Number of Vegetation Types = 16           # Number of different vegetation types

################ 1 VEGETATION 11 ##################################################
Vegetation Description   1 =  Open Water NLCD 2011 Class 11
Impervious Fraction      1 =  0.0
Overstory Present        1 =  FALSE        
Understory Present       1 =  FALSE            
Fractional Coverage      1 =  
Hemi Fract Coverage      1 = 
Trunk Space              1 =  
Aerodynamic Attenuation  1 =  
Radiation Attenuation    1 =  
Max Snow Int Capacity    1 =  
Snow Interception Eff    1 =  
Mass Release Drip Ratio  1 =  
Height                   1 =  
Overstory Monthly LAI    1 = 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
Understory Monthly LAI   1 = 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
Maximum Resistance       1 =  
Minimum Resistance       1 =  
Moisture Threshold       1 =  
Vapor Pressure Deficit   1 =  
Rpc                      1 =  
Overstory Monthly Alb    1 = 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Understory Monthly Alb   1 = 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Number of Root Zones     1 =  3 		   
Root Zone Depths         1 =  0.02 0.02 0.02
Overstory Root Fraction  1 =  0.20 0.40 0.40
Understory Root Fraction 1 =  0.20 0.40 0.40       



################ 2 VEGETATION 12 ##################################################

Vegetation Description   2 =  GLACIER Ice Snow NLCD 2011 Class 12
Impervious Fraction      2 =  0.0
Overstory Present        2 =  FALSE
Understory Present       2 =  FALSE
Fractional Coverage      2 = 
Trunk Space              2 = 
Aerodynamic Attenuation  2 = 
Radiation Attenuation    2 = 
Hemi Fract Coverage      2 = 
Clumping Factor          2 = 
Leaf Angle A             2 = 
Leaf Angle B             2 = 
Scattering Parameter     2 = 
Max Snow Int Capacity    2 = 
Mass Release Drip Ratio  2 = 
Snow Interception Eff    2 = 
Impervious Fraction      2 =  
Height                   2 =  
Overstory Monthly LAI    2 = 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
Understory Monthly LAI   2 = 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
Maximum Resistance       2 = 
Minimum Resistance       2 = 
Moisture Threshold       2 = 
Vapor Pressure Deficit   2 = 
Rpc                      2 = 
Overstory Monthly Alb    2 = 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
Understory Monthly Alb   2 =  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
Number of Root Zones     2 =  3
Root Zone Depths         2 =  0.02 0.02 0.02
Overstory Root Fraction  2 = 0.20 0.40 0.40
Understory Root Fraction 2 = 0.20 0.40 0.40

################ 3 VEGETATION 21 ##################################################

Vegetation Description   3 =  Developed Open Space(Grass/crop/shrub) NLCD 2011 Class 21
Impervious Fraction      3 =  0.0
Overstory Present        3 =  FALSE 
Understory Present       3 =  TRUE             
Fractional Coverage      3 =   
Hemi Fract Coverage      3 =  
Trunk Space              3 =  
Aerodynamic Attenuation  3 =  
Radiation Attenuation    3 =   
Max Snow Int Capacity    3 =  		  
Snow Interception Eff    3 =   
Mass Release Drip Ratio  3 =  
Height                   3 =  1.0
Overstory Monthly LAI    3 = 1.0 1.0 1.0 1.0 1.0 3.0 3.0 3.0 3.0 1.0 1.0 1.0
Understory Monthly LAI   3 = 1.0 1.0 1.0 1.0 1.0 3.0 3.0 3.0 3.0 1.0 1.0 1.0
Maximum Resistance       3 =  600.  
Minimum Resistance       3 =  280    
Moisture Threshold       3 =  0.33     
Vapor Pressure Deficit   3 =  4000     
Rpc                      3 =  .108
Overstory Monthly Alb    3= 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20
Understory Monthly Alb   3 = 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20
Number of Root Zones     3 = 3		   
Root Zone Depths         3 = 0.1 0.1 0.02  
Overstory Root Fraction  3 = 0.20 0.40 0.40
Understory Root Fraction 3 = 0.40 0.60 0.00    


################ 4 VEGETATION 22 ##################################################

Vegetation Description   4 =  Developed Low Intensity 20-49% Impervious NLCD 2011 Class 22
Impervious Fraction      4 =  0
IMPERVIOUS SURFACE ROUTING FILE = network/surface.routing.txt
Overstory Present        4 =  FALSE
Understory Present       4 =  TRUE   
Fractional Coverage      4 =   
Hemi Fract Coverage      4 =  
Trunk Space              4 =   
Aerodynamic Attenuation  4 =  
Radiation Attenuation    4 =  
Max Snow Int Capacity    4 =  
Snow Interception Eff    4 =  
Mass Release Drip Ratio  4 =  
Height                   4 =  0.2
Overstory Monthly LAI    4 = 1.0 1.0 1.0 1.0 1.0 3.0 3.0 3.0 3.0 1.0 1.0 1.0
Understory Monthly LAI   4 = 1.0 1.0 1.0 1.0 1.0 3.0 3.0 3.0 3.0 1.0 1.0 1.0
Maximum Resistance       4 =  3000.
Minimum Resistance       4 =  300
Moisture Threshold       4 =  0.33     
Vapor Pressure Deficit   4 =  4000    
Rpc                      4 =  0.108
Overstory Monthly Alb    4 = 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20
Understory Monthly Alb   4 = 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20
Number of Root Zones     4 =  3		   
Root Zone Depths         4 =  0.1 0.1 0.02  
Overstory Root Fraction  4 =  0.20 0.40 0.40
Understory Root Fraction 4 = 0.40 0.60 0.00   

################ 5 VEGETATION 23 ##################################################

Vegetation Description   5 =  Developed Medium Intensity 50-79% Impervious NLCD 2011 Class 23
Impervious Fraction      5 =  0
IMPERVIOUS SURFACE ROUTING FILE = network/surface.routing.txt
Overstory Present        5 =  FALSE
Understory Present       5 =  TRUE   
Fractional Coverage      5 =   
Hemi Fract Coverage      5 =  
Trunk Space              5 =   
Aerodynamic Attenuation  5 =  
Radiation Attenuation    5 =  
Max Snow Int Capacity    5 =  
Snow Interception Eff    5 =  
Mass Release Drip Ratio  5 =  
Height                   5 =  0.2
Overstory Monthly LAI    5 = 1.0 1.0 1.0 1.0 1.0 3.0 3.0 3.0 3.0 1.0 1.0 1.0
Understory Monthly LAI   5 = 1.0 1.0 1.0 1.0 1.0 3.0 3.0 3.0 3.0 1.0 1.0 1.0
Maximum Resistance       5 =  3000.
Minimum Resistance       5 =  300
Moisture Threshold       5 =  0.33     
Vapor Pressure Deficit   5 =  4000    
Rpc                      5 =  0.108
Overstory Monthly Alb    5 = 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20
Understory Monthly Alb   5 = 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20
Number of Root Zones     5 =  3		
Root Zone Depths         5 =  0.1 0.1 0.02 
Overstory Root Fraction  5 =  0.20 0.40 0.40
Understory Root Fraction 5 = 0.40 0.60 0.00   

################ 6 VEGETATION 24 ##################################################

Vegetation Description   6 =  Developed High Intensity 80-100% Impervious NLCD 2011 Class 24
Impervious Fraction      6 =  0
IMPERVIOUS SURFACE ROUTING FILE = ./network/surface.routing.txt
Overstory Present        6 =  FALSE  
Understory Present       6 =  TRUE    
Fractional Coverage      6 =  	   
Hemi Fract Coverage      6 =  
Trunk Space              6 =         
Aerodynamic Attenuation  6 =         
Radiation Attenuation    6 =         
Max Snow Int Capacity    6 =   
Snow Interception Eff    6 =  
Mass Release Drip Ratio  6 =  
Height                   6 =  0.2
Overstory Monthly LAI    6 = 1.0 1.0 1.0 1.0 1.0 3.0 3.0 3.0 3.0 1.0 1.0 1.0
Understory Monthly LAI   6 = 1.0 1.0 1.0 1.0 1.0 3.0 3.0 3.0 3.0 1.0 1.0 1.0
Maximum Resistance       6 =  3000.  
Minimum Resistance       6 =  340.   
Moisture Threshold       6 =  0.33 
Vapor Pressure Deficit   6 =  4000 
Rpc                      6 =  .108
Overstory Monthly Alb    6 = 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20
Understory Monthly Alb   6 = 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20 0.20
Number of Root Zones     6 = 3		   
Root Zone Depths         6 = 0.1 0.1 0.02
Overstory Root Fraction  6 = 0.20 0.40 0.40
Understory Root Fraction 6 = 0.40 0.60 0.00  



################ 7 VEGETATION 31 ##################################################

Vegetation Description   7 =  Barren Land (Rock/Sand/Clay) NLCD 2011 Class 31
Impervious Fraction      7 =  0
Overstory Present        7 =  FALSE 
Understory Present       7 =  FALSE   
Fractional Coverage      7 =  	   
Hemi Fract Coverage      7 =  
Trunk Space              7 =   
Aerodynamic Attenuation  7 =  
Radiation Attenuation    7 =   
Max Snow Int Capacity    7 =   
Snow Interception Eff    7 =  
Mass Release Drip Ratio  7 =  
Height                   7 =  
Overstory Monthly LAI    7 = 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
Understory Monthly LAI   7 = 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
Maximum Resistance       7 =  
Minimum Resistance       7 =  
Moisture Threshold       7 =  
Vapor Pressure Deficit   7 =  
Rpc                      7 =  
Overstory Monthly Alb    7 = 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Understory Monthly Alb   7 = 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Number of Root Zones     7 =  3 		   
Root Zone Depths         7 =  0.1 0.1 0.02
Overstory Root Fraction  7 =  0.20 0.40 0.40
Understory Root Fraction 7 =  0.20 0.40 0.40 

################ 8 VEGETATION 41 ##################################################
Vegetation Description   8 =  Deciduous Forest NLCD 2011 Class 41
Impervious Fraction      8 =  0.0
Detention Fraction       8 =  0
Detention Decay          8 =  0
Overstory Present        8 =  TRUE       
Understory Present       8 =  TRUE             
Fractional Coverage      8 =  0.73         
Hemi Fract Coverage      8 =  0.6
Clumping Factor          8 =  2.0
Leaf Angle A             8 =  0.5
Leaf Angle B             8 =  0.0
Scattering Parameter     8 =  0.8
Trunk Space              8 =  0.4           
Aerodynamic Attenuation  8 =  0.5          
Radiation Attenuation    8 =  0.2         
Max Snow Int Capacity    8 =  0.003		  
Snow Interception Eff    8 =  0.6              
Mass Release Drip Ratio  8 =  0.4             
Height                   8 =  30.83 0.5      
Overstory Monthly LAI    8 =  0.138 0.150 0.162 0.338 1.487 4.425 5.625 5.850 4.588 1.662 0.288 0.225
Understory Monthly LAI   8 =  0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5
Maximum Resistance       8 =  5000. 600.  
Minimum Resistance       8 =  250. 200.
Moisture Threshold       8 =  0.33 0.13     
Vapor Pressure Deficit   8 =  4000 4000   
Rpc                      8 =  .108 .108
Overstory Monthly Alb    8 =  0.16 0.16 0.14 0.12 0.09 0.07 0.07 0.07 0.10 0.13 0.16 0.16
Understory Monthly Alb   8 =  0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2
Number of Root Zones     8 =  3 		   
Root Zone Depths         8 =  0.1 0.1 0.02    
Overstory Root Fraction  8 =  0.20 0.40 0.40     
Understory Root Fraction 8 = 0.40 0.60 0.00   

################ 9 VEGETATION 42 ##################################################

Vegetation Description   9 =  Evergreen Forest
Impervious Fraction      9 =  0.0
Detention Fraction       9 =  0
Detention Decay          9 =  0
Overstory Present        9 =  TRUE        
Understory Present       9 =  TRUE             
Fractional Coverage      9 =  0.8         
Hemi Fract Coverage      9 =  0.7
Clumping Factor          9 =  4.0
Leaf Angle A             9 =  0.5
Leaf Angle B             9 =  0.0
Scattering Parameter     9 =  0.8
Trunk Space              9 =  0.5         
Aerodynamic Attenuation  9 =  2.0     
Radiation Attenuation    9 =  0.15         
Max Snow Int Capacity    9 =  0.03 
Snow Interception Eff    9 =  0.6              
Mass Release Drip Ratio  9 =  0.4             
Height                   9 =  35.01 0.5
Overstory Monthly LAI    9 =  5.812 5.812 5.812 5.812 5.812 5.812 5.812 5.812 5.812 5.812 5.812 5.812
Understory Monthly LAI   9 =  0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5
Maximum Resistance       9 =  5000. 3000.
Minimum Resistance       9 =  250. 200.   
Moisture Threshold       9 =  0.33 0.13     
Vapor Pressure Deficit   9 =  4000 4000   
Rpc                      9 =  .108 .108
Overstory Monthly Alb    9 =  0.16 0.16 0.14 0.12 0.09 0.06 0.06 0.06 0.10 0.13 0.16 0.16
Understory Monthly Alb   9 =  0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2
Number of Root Zones     9 =  3 		   
Root Zone Depths         9 =  0.1 0.1 0.02   
Overstory Root Fraction  9 =  0.20 0.40 0.40
Understory Root Fraction 9 = 0.40 0.60 0.00  

################ 10 VEGETATION 43 ##################################################
Vegetation Description   10 = Mixed Forest NLCD Class 43
Impervious Fraction      10 =   0.0
Detention Fraction       10 =   0
Detention Decay          10 =   0
Overstory Present        10 =   TRUE       
Understory Present       10 =   TRUE             
Fractional Coverage      10 =   0.73         
Hemi Fract Coverage      10 =   0.6
Clumping Factor          10 =   2.0
Leaf Angle A             10 =   0.5
Leaf Angle B             10 =   0.0
Scattering Parameter     10 =   0.8
Trunk Space              10 =   0.4           
Aerodynamic Attenuation  10 =   0.5          
Radiation Attenuation    10 =   0.2         
Max Snow Int Capacity    10 =   0.003		  
Snow Interception Eff    10 =   0.6              
Mass Release Drip Ratio  10 =   0.4             
Height                   10 =   35.83 0.5      
Overstory Monthly LAI    10 =   0.138 0.150 0.162 0.338 1.487 4.425 5.625 5.850 4.588 1.662 0.288 0.225
Understory Monthly LAI   10 =   0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5
Maximum Resistance       10 =   5000. 600.  
Minimum Resistance       10 =   250. 200.
Moisture Threshold       10 =   0.33 0.13     
Vapor Pressure Deficit   10 =   4000 4000   
Rpc                      10 =   .108 .108
Overstory Monthly Alb    10 =   0.16 0.16 0.14 0.12 0.09 0.07 0.07 0.07 0.10 0.13 0.16 0.16
Understory Monthly Alb   10 =   0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2
Number of Root Zones     10 =   3 		   
Root Zone Depths         10 =   0.1 0.1 0.02    
Overstory Root Fraction  10 =   0.20 0.40 0.40     
Understory Root Fraction 10 =   0.40 0.60 0.00   


################ 11 VEGETATION 52 ##################################################

Vegetation Description   11 =  Shrub/Scrub NLCD Class 52
Impervious Fraction      11 =  0.0
Detention Fraction       11 =  0.0
Detention Decay          11 =  0.0
Overstory Present        11 =  FALSE 
Understory Present       11 =  TRUE             
Fractional Coverage      11 =   
Hemi Fract Coverage      11 =  
Clumping Factor          11 = 
Leaf Angle A             11 = 
Leaf Angle B             11 = 
Scattering Parameter     11 = 
Trunk Space              11 =  
Aerodynamic Attenuation  11 =  
Radiation Attenuation    11 =   
Max Snow Int Capacity    11 =  		  
Snow Interception Eff    11 =   
Mass Release Drip Ratio  11 =  
Height                   11 =  2.0
Overstory Monthly LAI    11 =  
Understory Monthly LAI   11 =  0.58 0.63 0.63 0.63 0.92 1.77 2.55 2.55 1.73 0.97 0.73 0.63
Maximum Resistance       11 =  600.  
Minimum Resistance       11 =  200    
Moisture Threshold       11 =  0.33     
Vapor Pressure Deficit   11 =  4000
Rpc                      11 =  .108
Overstory Monthly Alb    11 =  
Understory Monthly Alb   11 =  0.22 0.22 0.18 0.16 0.14 0.10 0.10 0.10 0.15 0.17 0.22 0.22
Number of Root Zones     11 =  3 		   
Root Zone Depths         11 =  0.1 0.05 0.04   
Overstory Root Fraction  11 =  0.20 0.40 0.40      
Understory Root Fraction 11 = 0.40 0.60 0.00      

################ 12 VEGETATION 71 ##################################################

Vegetation Description   12 =  Grassland/Herbaceous NLCD Class 71
Impervious Fraction      12 =  0.0
Overstory Present        12 =  TRUE
Understory Present       12 =  TRUE             
Fractional Coverage      12 =  0.5
Hemi Fract Coverage      12 =  0.5 
Trunk Space              12 =  0.4
Aerodynamic Attenuation  12 =  0.3
Radiation Attenuation    12 =  0.1
Max Snow Int Capacity    12 =  0.003		  
Snow Interception Eff    12 =  0.6
Mass Release Drip Ratio  12 =  0.4
Height                   12 =  20.0 0.5      
Overstory Monthly LAI    12 =  2.0 2.0 2.0 2.0 2.0 6.0 6.0 6.0 6.0 2.0 2.0 2.0
Understory Monthly LAI   12 =  1.0 1.0 1.0 1.0 1.0 3.3 3.3 3.3 3.3 1.0 1.0 1.0
Maximum Resistance       12 =  1000 600.  
Minimum Resistance       12 =  280 200.    
Moisture Threshold       12 =  0.33 0.33     
Vapor Pressure Deficit   12 =  4000  4000    
Rpc                      12 =  0.108 0.108
Overstory Monthly Alb    12 =  0.15 0.15 0.15 0.15 0.15 0.14 0.14 0.14 0.14 0.15 0.15 0.15 
Understory Monthly Alb   12 =  0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 
Number of Root Zones     12 =  3 		   
Root Zone Depths         12 =  0.1 0.05 0.04
Overstory Root Fraction  12 =  0.20 0.40 0.40      
Understory Root Fraction 12 = 0.40 0.60 0.00      

################ 13 VEGETATION 81 ##################################################

Vegetation Description   13 =   Pasture/Hay NLCD Class 81
Impervious Fraction      13 =   0.0
Overstory Present        13 =   TRUE
Understory Present       13 =   TRUE             
Fractional Coverage      13 =   0.5
Hemi Fract Coverage      13 =   0.5 
Trunk Space              13 =   0.4
Aerodynamic Attenuation  13 =   0.3
Radiation Attenuation    13 =   0.1
Max Snow Int Capacity    13 =   0.003		  
Snow Interception Eff    13 =   0.6
Mass Release Drip Ratio  13 =   0.4
Height                   13 =   20.0 0.5      
Overstory Monthly LAI    13 =   2.0 2.0 2.0 2.0 2.0 6.0 6.0 6.0 6.0 2.0 2.0 2.0
Understory Monthly LAI   13 =   1.0 1.0 1.0 1.0 1.0 3.3 3.3 3.3 3.3 1.0 1.0 1.0
Maximum Resistance       13 =   1000 600.  
Minimum Resistance       13 =   280 200.    
Moisture Threshold       13 =   0.33 0.33     
Vapor Pressure Deficit   13 =   4000  4000    
Rpc                      13 =   0.108 0.108
Overstory Monthly Alb    13 =   0.15 0.15 0.15 0.15 0.15 0.14 0.14 0.14 0.14 0.15 0.15 0.15 
Understory Monthly Alb   13 =   0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 
Number of Root Zones     13 =   3 		   
Root Zone Depths         13 =   0.1 0.1 0.02
Overstory Root Fraction  13 =   0.20 0.40 0.40      
Understory Root Fraction 13 = 0.40 0.60 0.00     
   
################ 14 VEGETATION 82 ##################################################

Vegetation Description   14 =    Cultivated Crops NLCD Class 82
Impervious Fraction      14 =    0.0
Overstory Present        14 =    TRUE
Understory Present       14 =    TRUE             
Fractional Coverage      14 =    0.5
Hemi Fract Coverage      14 =    0.5 
Trunk Space              14 =    0.4
Aerodynamic Attenuation  14 =    0.3
Radiation Attenuation    14 =    0.1
Max Snow Int Capacity    14 =    0.003		  
Snow Interception Eff    14 =    0.6
Mass Release Drip Ratio  14 =    0.4
Height                   14 =    20.0 0.5      
Overstory Monthly LAI    14 =    2.0 2.0 2.0 2.0 2.0 6.0 6.0 6.0 6.0 2.0 2.0 2.0
Understory Monthly LAI   14 =    1.0 1.0 1.0 1.0 1.0 3.3 3.3 3.3 3.3 1.0 1.0 1.0
Maximum Resistance       14 =    1000 600.  
Minimum Resistance       14 =    280 200.    
Moisture Threshold       14 =    0.33 0.33     
Vapor Pressure Deficit   14 =    4000  4000    
Rpc                      14 =    0.108 0.108
Overstory Monthly Alb    14 =    0.15 0.15 0.15 0.15 0.15 0.14 0.14 0.14 0.14 0.15 0.15 0.15 
Understory Monthly Alb   14 =    0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 0.18 
Number of Root Zones     14 =    3 		   
Root Zone Depths         14 =    0.1 0.1 0.02
Overstory Root Fraction  14 =    0.20 0.40 0.40      
Understory Root Fraction 14 = 0.40 0.60 0.00    

################ 15 VEGETATION 90 ##################################################

Vegetation Description   15 =  Woody Wetlands  NLCD 2011 Class 90
Impervious Fraction      15 =  0.0
Overstory Present        15 =  FALSE        
Understory Present       15 =  TRUE           
Fractional Coverage      15 =  
Hemi Fract Coverage      15 = 
Trunk Space              15 =  
Aerodynamic Attenuation  15 =  
Radiation Attenuation    15 =  
Max Snow Int Capacity    15 =  
Snow Interception Eff    15 =  
Mass Release Drip Ratio  15 =  
Height                   15 =  0.5
Overstory Monthly LAI    15 =  
Understory Monthly LAI   15 =  0.5 0.5 0.5 0.5 0.5 6.0 6.0 6.0 6.0 0.5 0.5 0.5 
Maximum Resistance       15 =  600
Minimum Resistance       15 =  200
Moisture Threshold       15 =  0.33
Vapor Pressure Deficit   15 =  4000
Rpc                      15 =  0.108
Overstory Monthly Alb    15 =  
Understory Monthly Alb   15 =  0.19 0.19 0.19 0.17 0.17 0.16 0.16 0.16 0.16 0.17 0.19 0.19
Number of Root Zones     15 =  3		   
Root Zone Depths         15 =  0.1 0.05 0.04
Overstory Root Fraction  15 =  0.20 0.40 0.40
Understory Root Fraction 15 = 0.4 0.6 0.0  


################ 16 VEGETATION 95 ##################################################

Vegetation Description   16 =   Emergent Herbaceous Wetlands  NLCD 2011 Class 95
Impervious Fraction      16 =   0.0
Overstory Present        16 =   FALSE        
Understory Present       16 =   TRUE           
Fractional Coverage      16 =   
Hemi Fract Coverage      16 =  
Trunk Space              16 =   
Aerodynamic Attenuation  16 =   
Radiation Attenuation    16 =   
Max Snow Int Capacity    16 =   
Snow Interception Eff    16 =   
Mass Release Drip Ratio  16 =   
Height                   16 =   0.5
Overstory Monthly LAI    16 =   
Understory Monthly LAI   16 =   0.5 0.5 0.5 0.5 0.5 6.0 6.0 6.0 6.0 0.5 0.5 0.5 
Maximum Resistance       16 =   600
Minimum Resistance       16 =   200
Moisture Threshold       16 =   0.33
Vapor Pressure Deficit   16 =   4000
Rpc                      16 =   0.108
Overstory Monthly Alb    16 =   
Understory Monthly Alb   16 =   0.19 0.19 0.19 0.17 0.17 0.16 0.16 0.16 0.16 0.17 0.19 0.19
Number of Root Zones     16 =   3		   
Root Zone Depths         16 =   0.1 0.1 0.02
Overstory Root Fraction  16 =   0.20 0.40 0.40
Understory Root Fraction 16 = 0.4 0.6 0.0   
      

################################################################################
# MODEL OUTPUT SECTION
################################################################################
[OUTPUT]                                  # Information what to output when
Output Directory           = output/output1960/output_skagitHYDRO_bclivlow_G/
Initial State Directory    = modelstate/

################ PIXEL DUMPS ###################################################

Number of Output Pixels    = 2

# For each pixel make a key-entry pair as indicated below, varying the 
# number for the output pixel  (1, .. , Number of Output Pixel)

North Coordinate         1 = 5395456
East Coordinate          1 = 631866.1
Name                     1 = GorgeDam279m

North Coordinate         2 = 5378093.11 
East Coordinate          2 = 637828.77
Name                     2 = ElDorado2644m

################ MODEL STATE ###################################################                
Number of Model States     =  4            # Number of model states to dump

# For each model state make a key-entry pair as indicated below, varying the 
# number for the model state dump (1, .. , Number of Model States)

State Date               1 = 10/01/2006-00  # Time for model state dump
State Date               2 = 11/01/2006-00  # Time for model state dump
State Date               3 = 10/01/2009-00  # Time for model state dump
State Date               4 = 11/01/2009-00  # Time for model state dump

################ MODEL MAPS ####################################################
Number of Map Variables    = 3       # Number of different variables for
                                          # which you want to output maps

# For each of the variables make a block like the one that follows, varying
# the number of the variable (n = 1, .. , Number of Map Variables)

Map Variable             1 = 503            # 503= water table depth  ID of the variable to output
Map Layer                1 = 1             # If the variable exists for a number
					  # of layers, specify the layers here
					  # with the top layer = 1
Number of Maps           1 = 26            # Number of maps you would like to 
Map Date 1               1 = 08/01/1987-00   # output for this variable
Map Date 2               1 = 08/01/1988-00     # Vary the first number from 
Map Date 3               1 = 08/01/1989-00 
Map Date 4               1 = 08/01/1990-00 
Map Date 5               1 = 08/01/1991-00 
Map Date 6               1 = 08/01/1992-00 
Map Date 7               1 = 08/01/1993-00 
Map Date 8               1 = 08/01/1994-00 
Map Date 9               1 = 08/01/1995-00 
Map Date 10              1 = 08/01/1996-00 
Map Date 11              1 = 08/01/1997-00 
Map Date 12              1 = 08/01/1998-00 
Map Date 13              1 = 08/01/1999-00 
Map Date 14              1 = 08/01/2000-00 
Map Date 15              1 = 08/01/2001-00 
Map Date 16              1 = 08/01/2002-00 
Map Date 17              1 = 08/01/2003-00 
Map Date 18              1 = 08/01/2004-00 
Map Date 19              1 = 08/01/2005-00 
Map Date 20              1 = 08/01/2006-00 
Map Date 21              1 = 08/01/2007-00 
Map Date 22              1 = 08/01/2008-00 
Map Date 23              1 = 08/01/2009-00 
Map Date 24              1 = 08/01/2010-00
Map Date 25              1 = 08/01/2011-00 
Map Date 26              1 = 08/01/2012-00 


Map Variable             2 = 404            # 404 = SWE   ID of the variable to output
Map Layer                2 = 1             # If the variable exists for a number
					  # of layers, specify the layers here
					  # with the top layer = 1
Number of Maps           2 = 26           # Number of maps you would like to 
Map Date 1               2 = 04/01/1987-00   # output for this variable
Map Date 2               2 = 04/01/1988-00     # Vary the first number from 
Map Date 3               2 = 04/01/1989-00 
Map Date 4               2 = 04/01/1990-00 
Map Date 5               2 = 04/01/1991-00 
Map Date 6               2 = 04/01/1992-00 
Map Date 7               2 = 04/01/1993-00 
Map Date 8               2 = 04/01/1994-00 
Map Date 9               2 = 04/01/1995-00 
Map Date 10              2 = 04/01/1996-00 
Map Date 11              2 = 04/01/1997-00 
Map Date 12              2 = 04/01/1998-00 
Map Date 13              2 = 04/01/1999-00 
Map Date 14              2 = 04/01/2000-00 
Map Date 15              2 = 04/01/2001-00 
Map Date 16              2 = 04/01/2002-00 
Map Date 17              2 = 04/01/2003-00 
Map Date 18              2 = 04/01/2004-00 
Map Date 19              2 = 04/01/2005-00 
Map Date 20              2 = 04/01/2006-00 
Map Date 21              2 = 04/01/2007-00 
Map Date 22              2 = 04/01/2008-00 
Map Date 23              2 = 04/01/2009-00 
Map Date 24              2 = 04/01/2010-00
Map Date 25              2 = 04/01/2011-00 
Map Date 26              2 = 04/01/2012-00 


Map Variable             3 = 705            # 705= Ice    ID of the variable to output
Map Layer                3 = 1             # If the variable exists for a number
					  # of layers, specify the layers here
					  # with the top layer = 1
Number of Maps           3 = 26           # Number of maps you would like to 
Map Date 1               3 = 09/30/1987-00   # output for this variable
Map Date 2               3 = 09/30/1988-00     # Vary the first number from 
Map Date 3               3 = 09/30/1989-00 
Map Date 4               3 = 09/30/1990-00 
Map Date 5               3 = 09/30/1991-00 
Map Date 6               3 = 09/30/1992-00 
Map Date 7               3 = 09/30/1993-00 
Map Date 8               3 = 09/30/1994-00 
Map Date 9               3 = 09/30/1995-00 
Map Date 10              3 = 09/30/1996-00 
Map Date 11              3 = 09/30/1997-00 
Map Date 12              3 = 09/30/1998-00 
Map Date 13              3 = 09/30/1999-00 
Map Date 14              3 = 09/30/2000-00 
Map Date 15              3 = 09/30/2001-00 
Map Date 16              3 = 09/30/2002-00 
Map Date 17              3 = 09/30/2003-00 
Map Date 18              3 = 09/30/2004-00 
Map Date 19              3 = 09/30/2005-00 
Map Date 20              3 = 09/30/2006-00 
Map Date 21              3 = 09/30/2007-00 
Map Date 22              3 = 09/30/2008-00 
Map Date 23              3 = 09/30/2009-00 
Map Date 24              3 = 09/30/2010-00
Map Date 25              3 = 09/30/2011-00 
Map Date 26              3 = 09/30/2012-00 

#Map Date 1              1 = 10/01/1960-00
#Map Date 2              1 = 10/01/1965-00
#Map Date 3              1 = 10/01/1970-00
#Map Date 4              1 = 10/01/1975-00
#Map Date 5              1 = 10/01/1980-00
#Map Date 6              1 = 10/01/1985-00
#Map Date 7              1 = 10/01/1990-00
#Map Date 8              1 = 10/01/1995-00
#Map Date 9              1 = 10/01/2000-00
#Map Date 10             1 = 10/01/2005-00
#Map Date 11             1 = 01/01/2010-00
#Map Date 12             1 = 10/01/2010-00
#Map Date 13             1 = 10/01/2015-00

################ MODEL IMAGES ##################################################

Number of Image Variables  = 0            # Number of variables for which you 
                                          # would like to output images

# For each of the variables make a block like the one that follows, varying
# the number of the variable (n = 1, .. , Number of Image Variables)

Image Variable        1 = 501                # ID of the variable to output
Image Layer           1 = 1                # If the variable exists for a number
					  # of layers, specify the layers here
					  # with the top layer = 1
Image Start           1 =                 # First timestep for which to output
                                          # an image
Image End             1 =                 # Last timestep for which to output
                                          # an image
Image Interval        1 =                 # Time interval between images (hours)
Image Upper Limit     1 =                 # All values in the output equal to or
                                          # greater than this limit will be set 
                                          # to 255
Image Lower Limit     1 =                 # All values in the output equal to or
                                          # smaller than this limit will be set 
                                          # to 0
################ GRAPHIC IMAGES ##################################################

Number of Graphics      =  0               # Number of variables for which you 
                                          # would like to output images
Graphics ID           1 = 19                # ID of the variable to output
Graphics ID           1 = 22
Graphics ID           2 = 23
Graphics ID           3 = 24
Graphics ID           4 = 25
Graphics ID           5 = 43
Graphics ID           6 = 44
Graphics ID           7 = 8
Graphics ID           8 = 2
Graphics ID           9 = 50
Graphics ID          10 = 1

# 1	SWE (mm)
# 2	Water Table Depth (mm)
# 3	Digital Elevation Model (m)
# 4	Vegetation Class (index #)
# 5	Soil Class (index #)
# 6	Soil Depth (mm)
# 7	Precipitation at current time step (mm/time step)
# 8	Incoming Shortwave (Beam and Diffuse) (W/sqm)
# 9	Intercepted Snow (mm)
# 10	Snow Surface Temp (C)
# 11	Cold Content of snow entire snow pack (kJ)
# 12	Snow Melt (as Outflow minus Precip, can be negative) (mm/time step)
# 13	Snow Pack Outflow (mm/time step)
# 14	Saturated Subsurface Flow (mm/time step)
# 15	Overland Flow(mm)
# 16	Total Evapotranspiration (soil + all veg layers)
# 17	Ground Snow pack vapor flux (mm)
# 18	Intercepted snow pack vapor flux (mm)
# 19    Soil Moisture (Surface Layer) % of saturation (i.e. porosity)
# 20    Soil Moisture (2nd Layer) % of saturation (i.e. porosity)
# 21    Soil Moisture (3rd Layer) % of saturation (i.e. porosity)
# 22    Accumulated Precip (mm)
# 23    air temperature (C)
# 24    wind speed (m/s)
# 25    relative humidity
# 26    Prism Precip Field (mm)
# 31    Overstory Transpiration (mm)
# 32    Understory Transpirtation (mm)
# 33    Soil Evaporation (mm)
# 34    Overstory Evaporation (mm)
# 35    Understory Evaportation (mm)
# 41    Sky View Factor (%)
# 42    Shade Map (%)
# 43    Direct Beam Shortwave Rad (W/sqm)
# 44    Diffuse Beam Shortwave Rad (W/sqm)
# 45    Aspect (degrees)
# 46    Slope (percent)
# 50    Channel Subsurface Interception (mm)
# 51    Road Subsurface Interception (mm)
# WARNING Use soil mositure layers with caution, to minimize calculations during redraw
# DHSVM does not check to make sure that the assigned soil layer exists

################################################################################
# END OF INPUT FILE
################################################################################
[End]                                     # This is probably not needed, but 
                                          # just in case (to close the previous
                                          # section)

