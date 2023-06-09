import matplotlib as mpl
import matplotlib.pyplot as plt
import contextily as ctx

# For reproducibility
import random
import numpy as np
import warnings
from pysal.lib import weights
warnings.filterwarnings("ignore")

r_state = 42
random.seed(r_state)
np.random.seed(r_state)
#%%
import pandas as pd
import pysal as ps
import geopandas as gpd
import requests
import glob
import re
import os
import io
import zipfile

from io import BytesIO

#%%
# Create folders
lkp = os.path.join('data','lkp')
shp = os.path.join('data','shp')

for d in [lkp,shp]:
    if not os.path.exists(d):
        os.makedirs(d)

shps = os.path.join(shp, 'src')
if not os.path.exists(shps):
    os.makedirs(shps)

#%%
boroughs = ['City of London','Barking and Dagenham','Barnet','Bexley','Brent','Bromley',
            'Camden','Croydon','Ealing','Enfield','Greenwich','Hackney','Hammersmith and Fulham',
            'Haringey','Harrow','Havering','Hillingdon','Hounslow','Islington',
            'Kensington and Chelsea','Kingston upon Thames','Lambeth','Lewisham',
            'Merton','Newham','Redbridge','Richmond upon Thames','Southwark','Sutton',
            'Tower Hamlets','Waltham Forest','Wandsworth','Westminster']

#%% Regions
# https://hub.arcgis.com/datasets/58247b4f427443d381d7d82e3a7565e1_0/explore
# Use glob to find .shp files in the folder
regions = glob.glob(shps + '/Regions*/*.shp')[0]
#regions = glob.glob(shpt+'/previous*/*.shp')[0]

print("Processing: " + regions)
regions = gpd.read_file(regions)

london  = regions[regions.rgn16nm=='London']
london.reset_index(inplace=True, drop=True)
#london.crs = {'init':'epsg:4326'}
#london = london.to_crs(epsg=27700)
london.to_file(os.path.join(shp,'London.shp'))
print("Done.")
#%% Regions
# Plot the London shapefile
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the London shapefile
london.plot(ax=ax, alpha=0.5,edgecolor='k')
#Plot basemap
ctx.add_basemap(ax, crs=london.crs.to_string() )
# Add a title and show the plot
plt.title('London Shapefile with Basemap')
plt.show()
#%% GLA
# https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london

counties = glob.glob(os.path.join(shps, 'statistical-gis-boundaries-london', 'ESRI', '*Borough*.shp'))[0]

print("Processing: " + counties)
LAs = gpd.read_file(counties)

LAs = LAs.loc[LAs.NAME.isin(boroughs)].reset_index(drop=True)
LAs.crs = {'init': u'epsg:27700'}
#LAs = LAs.to_crs({'init':'epsg:27700'})

print("\tSaving to shapefile...")
LAs.to_file(os.path.join(shp,'Boroughs.shp'))

print("Done.")

# Plot the London shapefile
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the London shapefile
LAs.plot(ax=ax, alpha=0.5,edgecolor='k')
#Plot basemap
ctx.add_basemap(ax, crs=london.crs.to_string(),source=ctx.providers.CartoDB.Positron )
# Add a title and show the plot
plt.title('London Shapefile with Basemap')
plt.show()



#%% lsoas 2011
# https://data.london.gov.uk/dataset/lsoa-atlas
lsoas = glob.glob(os.path.join(shps, 'statistical-gis-boundaries-london', 'ESRI', '*LSOA*.shp'))

for l in lsoas:
    print("Processing: " + l)
    lsoa_y = gpd.read_file(l)

    # Extract the year as 4 digits
    m     = re.search(r'\d{4}',l)
    lyear = l[m.start():m.end()]

    # Set projection
    lsoa_y.crs = {'init':'epsg:27700'}

    # Common name
    lsoa_y.insert(0, 'lsoacd',
                  lsoa_y[[x for x in lsoa_y.columns if 'LSOA' in x and ('CD' in x or 'CODE' in x)][0]])

    print("\tSaving to shapefile...")
    lsoa_y.to_file(os.path.join(shp,'LSOAs ' + str(lyear) + '.shp'))

    print("\tSaving to pickle...")
    lsoa_y.to_pickle(os.path.join(lkp,'LSOAs ' + str(lyear) + '.pkl'))

print("Done.")

# Plot the London shapefile
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the London shapefile
lsoa_y.plot(ax=ax, alpha=0.5,edgecolor='k')
#Plot basemap
ctx.add_basemap(ax, crs=london.crs.to_string(),source=ctx.providers.CartoDB.Positron )
# Add a title and show the plot
plt.title('London Shapefile with Basemap')
plt.show()

#%% 2011
lsoag  = glob.glob(os.path.join(shps, '*LSOA_Dec_2011*/*.shp'))[0] # LSOAs Generalised

print("Processing: " + lsoag)
lsoa_g = gpd.read_file(lsoag)

lsoa_g['borough'] = lsoa_g.LSOA11NM.str.extract("^(.+?) [0-9A-Z]{4}") # Extract borough/council names for subsetting
lsoa_g = lsoa_g[lsoa_g.borough.isin(boroughs)] # And select only those boroughs that match the array create above
print("\tExtracted " + str(len(lsoa_g)) + " London LSOAs.")

print("\tSaving to shapefile...")
lsoa_g.to_file(os.path.join(shp,'LSOA-2011-Weights.shp'))

w = weights.Queen.from_shapefile(os.path.join(shp,'LSOA-2011-Weights.shp'))
print("Check these results against above information:")
print("\tMean neighbours: " + str(w.mean_neighbors))
print("\tMax neighbours:  " + str(w.max_neighbors))
print("\tMin neighbours:  " + str(w.min_neighbors))
print("\tNo. Islands:     " + str(len(w.islands)))


#%% 2021
lsoag2021  = glob.glob(os.path.join(shps, '*LSOA_Dec_2021*/*.shp'))[0] # LSOAs Generalised

print("Processing: " + lsoag2021)
lsoa_g2021 = gpd.read_file(lsoag2021)

lsoa_g2021['borough'] = lsoa_g2021.LSOA21NM.str.extract("^(.+?) [0-9A-Z]{4}") # Extract borough/council names for subsetting
lsoa_g2021 = lsoa_g2021[lsoa_g2021.borough.isin(boroughs)] # And select only those boroughs that match the array create above
print("\tExtracted " + str(len(lsoa_g2021)) + " London LSOAs.")

print("\tSaving to shapefile...")
lsoa_g.to_file(os.path.join(shp,'LSOA-2021-Weights.shp'))

w = weights.Queen.from_shapefile(os.path.join(shp,'LSOA-2021-Weights.shp'))
print("Check these results against above information:")
print("\tMean neighbours: " + str(w.mean_neighbors))
print("\tMax neighbours:  " + str(w.max_neighbors))
print("\tMin neighbours:  " + str(w.min_neighbors))
print("\tNo. Islands:     " + str(len(w.islands)))


# Set projection
lsoa_g2021.crs = {'init':'epsg:27700'}

# Common name
lsoa_g2021.insert(0, 'lsoacd',
                  lsoa_g2021[[x for x in lsoa_g2021.columns if 'LSOA' in x and ('CD' in x or 'CODE' in x)][0]])

print("\tSaving to shapefile...")
lsoa_g2021.to_file(os.path.join(shp,'LSOAs 2021' + '.shp'))

print("\tSaving to pickle...")
lsoa_g2021.to_pickle(os.path.join(lkp,'LSOAs 2021' + '.pkl'))
#%%
wards = glob.glob(os.path.join(shps, 'statistical-gis-boundaries-london', 'ESRI', '*Ward*Merged.shp'))[0]

print("Processing wards...")
ward_geo = gpd.read_file(wards)
ward_geo.crs = {'init':'epsg:27700'}

print("\tSaving to shapefile...")
ward_geo.to_file(os.path.join(shp,'Wards.shp'))

# Plot the London shapefile
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the London shapefile
ward_geo.plot(ax=ax, alpha=0.5,edgecolor='k')
#Plot basemap
ctx.add_basemap(ax, crs=london.crs.to_string(),source=ctx.providers.CartoDB.Positron )
# Add a title and show the plot
plt.title('London Shapefile with Basemap')
plt.show()
#%%
# Create a mapping for LSOAs to Wards
lsoa = gpd.read_file(os.path.join(shp,'LSOAs 2011.shp'))
lsoa.crs = {'init':'epsg:27700'}

lsoa_c = lsoa
lsoa_c.geometry = lsoa_c.centroid
lsoa_c.to_file(os.path.join(shp,'LSOAs 2011 Points.shp'))

print("\tJoining Wards to LSOAs...")
t = gpd.sjoin(lsoa_c, ward_geo, how='left')
t.rename(columns={
    'GSS_CODE':'gss_cd',
    'LB_GSS_CD':'lb_gss_cd'
}, inplace=True)
t[['lsoacd','gss_cd','lb_gss_cd']].to_csv(os.path.join(lkp,'LSOA_WARD_JR.csv'), index=False)

print("Done.")

#%%
oas = glob.glob(os.path.join(shps, 'statistical-gis-boundaries-london', 'ESRI', 'OA_*.shp'))[0]

print("Processing Output Areas...")
oa_geo = gpd.read_file(oas)
oa_geo.crs = {'init':'epsg:27700'}

print("\tSaving to shapefile...")
oa_geo.to_file(os.path.join(shp,'OAs 2011.shp'))

# Create a mapping for LSOAs to Wards
lsoa = gpd.read_file(os.path.join(shp,'LSOAs 2011.shp'))
lsoa.crs = {'init':'epsg:27700'}

oa_c = oa_geo
oa_c.geometry = oa_c.centroid

print("\tSaving point OAs...")
oa_c.to_file(os.path.join(shp,'OAs 2011 Points.shp'))

print("\tOutput Areas to LSOAs...")

oa_geo.rename(columns={
    'LSOA11CD':'lsoacd',
    'OA11CD':'oacd'
}, inplace=True)
oa_geo[['lsoacd','oacd']].to_csv(os.path.join(lkp,'LSOA_OA_JR.csv'), index=False)

print("Done.")

#%%
import shutil
shutil.rmtree(shps)
print("Done.")