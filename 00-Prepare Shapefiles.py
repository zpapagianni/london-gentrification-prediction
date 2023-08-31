#%
# Define random state to reproduce
r_state = 58
random.seed(r_state)
np.random.seed(r_state)

# Load libraries
import matplotlib.pyplot as plt
import contextily as ctx
import random
import numpy as np
import warnings
from pysal.lib import weights
import pandas as pd
import pysal as ps
import geopandas as gpd
import glob
import re
import os
import io
import zipfile
from scipy.stats import gmean
import shutil

warnings.filterwarnings("ignore")

from io import BytesIO
#%%
# Create folders path
lkp = os.path.join('data', 'lkp')
shp = os.path.join('data', 'shp')
src = os.path.join('data','src')
#%%
# Reset and Delete old folders
if os.path.exists(lkp):
    shutil.rmtree(lkp)
if os.path.exists(shp):
    shutil.rmtree(shp)
if os.path.exists(src):
    shutil.rmtree(src)
print("Done.")
#%%
# Specify input data path data/input/zip_folder
zip_folder = os.path.join('data','input', 'zip')

# Create folders
for d in [lkp, shp,src]:
    if not os.path.exists(d):
        os.makedirs(d)

# Create temporary folder to extract zip files
shpt = os.path.join(shp, 'temp')
if not os.path.exists(shpt):
    os.makedirs(shpt)
#%%
# Define boroughs
boroughs = ['City of London','Barking and Dagenham','Barnet','Bexley','Brent','Bromley',
            'Camden','Croydon','Ealing','Enfield','Greenwich','Hackney','Hammersmith and Fulham',
            'Haringey','Harrow','Havering','Hillingdon','Hounslow','Islington',
            'Kensington and Chelsea','Kingston upon Thames','Lambeth','Lewisham',
            'Merton','Newham','Redbridge','Richmond upon Thames','Southwark','Sutton',
            'Tower Hamlets','Waltham Forest','Wandsworth','Westminster']

# Create a dictionary of borough names in 2018 and their corresponding 2011 names
borough_rename = {
    'City of Westminster': 'Westminster',
    'City and County of the City of London': 'City of London'}
#%%
# Extract zip files
# Iterate over each file in the folder
if len(os.listdir(shpt)) == 0:
    for file_name in os.listdir(zip_folder):
        if file_name.endswith('.zip'):  # Check if the file is a zip file
            file_path = os.path.join(zip_folder, file_name)  # Full path of the zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(shpt)
    print("Done.")
else:
    print("The folder already contains the extracted files .")


# Use glob to find .shp files in the folder
regions = glob.glob(shpt + '/Regions*.shp')[0]
print("Processing: " + regions)

# Read the shapefile into a GeoDataFrame
regions = gpd.read_file(regions)

# Select the region for London
london = regions[regions.rgn16nm=='London']

# Reset the index of the GeoDataFrame
london.reset_index(inplace=True, drop=True)

# Save the London shapefile
london.to_file(os.path.join(shp,'London.shp'))
print("Done.")

#%%
# Plot the London shapefile
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the London shapefile
london.plot(ax=ax, alpha=0.5,edgecolor='k')
#Plot basemap
ctx.add_basemap(ax, crs=london.crs.to_string(),source=ctx.providers.CartoDB.Positron)
plt.axis('off')
# Add a title and show the plot
plt.title('London Geographic Boundary')
plt.show()

#%%
# Use glob to find the Borough shapefile in the specified directory
counties = glob.glob(os.path.join(shpt, 'statistical-gis-boundaries-london', 'ESRI', '*Borough*.shp'))[0]
print("Processing: " + counties)

# Read the shapefile into a GeoDataFrame
brh = gpd.read_file(counties)

# Choose London boroughs by filtering the GeoDataFrame based on the 'NAME' column
brh = brh.loc[brh.NAME.isin(boroughs)].reset_index(drop=True)

# Set the coordinate reference system (CRS) of the GeoDataFrame
brh.crs = {'init': u'epsg:27700'}

# Save the borough shapefile
print("\tSaving to shapefile...")
brh.to_file(os.path.join(shp, 'Boroughs.shp'))

print("Done.")

#%%
# Plot the London shapefile
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the London shapefile
brh.plot(ax=ax, alpha=0.5, edgecolor='k')
#Plot basemap
ctx.add_basemap(ax, crs=london.crs.to_string(), source=ctx.providers.CartoDB.Positron)
plt.axis('off')
# Add a title and show the plot
plt.title('London Boroughs')
plt.show()
#%%
lsoas = glob.glob(os.path.join(shpt, 'statistical-gis-boundaries-london', 'ESRI', '*LSOA*.shp'))

for l in lsoas:
    print("Processing: " + l)
    lsoa_y = gpd.read_file(l)

    # Extract the year as 4 digits
    m = re.search(r'\d{4}', l)
    lyear = l[m.start():m.end()]

    # Check if the year is 2011
    if lyear=='2011':
        # Set projection
        lsoa_y.crs = {'init': 'epsg:27700'}

        # Common name
        lsoa_y.insert(0, 'lsoacd',
                      lsoa_y[[x for x in lsoa_y.columns if 'LSOA' in x and ('CD' in x or 'CODE' in x)][0]])

        print("\tSaving to shapefile...")
        lsoa_y.to_file(os.path.join(shp, 'LSOAs ' + str(lyear) + '.shp'))

        print("\tSaving to pickle...")
        lsoa_y.to_pickle(os.path.join(lkp, 'LSOAs ' + str(lyear) + '.pkl'))


print("Done.")

#%%
# Plot the London shapefile
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the London shapefile
lsoa_y.plot(ax=ax, alpha=0.5, edgecolor='k')
# Plot basemap
ctx.add_basemap(ax, crs=london.crs.to_string(), source=ctx.providers.CartoDB.Positron)
# Add a title and show the plot
plt.title('London LSOAs 2011')
plt.show()

#%%
lsoag = glob.glob(os.path.join(shpt, '*LSOA_2011*.shp'))[0]  # LSOAs Generalised

print("Processing: " + lsoag)
lsoa_g = gpd.read_file(lsoag)

lsoa_g['borough'] = lsoa_g.LSOA11NM.str.extract("^(.+?) [0-9A-Z]{4}")  #  Extract borough/council names for subsetting
lsoa_g = lsoa_g[lsoa_g.borough.isin(boroughs)]  # And select only those boroughs that match the array create above
print("\tExtracted " + str(len(lsoa_g)) + " London LSOAs.")

print("\tSaving to shapefile...")
lsoa_g.to_file(os.path.join(shp, 'LSOA-2011-Weights.shp'))

w = weights.Queen.from_shapefile(os.path.join(shp, 'LSOA-2011-Weights.shp'))
print("Check results:")
print("\tMean neighbours: " + str(w.mean_neighbors))
print("\tMax neighbours:  " + str(w.max_neighbors))
print("\tMin neighbours:  " + str(w.min_neighbors))
print("\tNo. Islands:     " + str(len(w.islands)))

#%% Selecting and Joining Wards for 2011
wards2011 = glob.glob(os.path.join(shpt, 'statistical-gis-boundaries-london', 'ESRI', '*Ward*Merged.shp'))[0]

print("Processing wards...")
ward_geo2011 = gpd.read_file(wards2011)
ward_geo2011.crs = {'init': 'epsg:27700'}

print("\tSaving to shapefile...")
ward_geo2011.to_file(os.path.join(shp, 'Wards 2011.shp'))
#%%
# Plot the London shapefile
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the London shapefile
ward_geo2011.plot(ax=ax, alpha=0.5, edgecolor='k')
#Plot basemap
ctx.add_basemap(ax, crs=london.crs.to_string(),source=ctx.providers.CartoDB.Positron )
# Add a title and show the plot
plt.title('London Wards in 2011')
plt.show()

#%%
# Read the LSOAs 2011 shapefile into a GeoDataFrame
print("Processing LSOAs...")
lsoa2011 = gpd.read_file(os.path.join(shp,'LSOAs 2011.shp'))

# Set the coordinate reference system (CRS) of the GeoDataFrame
lsoa2011.crs = {'init':'epsg:27700'}

# Create a copy of the LSOAs 2011 GeoDataFrame
lsoa_c2011 = lsoa2011

# Convert the geometry of the LSOAs to their centroids
lsoa_c2011.geometry = lsoa_c2011.centroid

# Save the LSOAs 2011 centroids as a new shapefile
lsoa_c2011.to_file(os.path.join(shp,'LSOAs 2011 Points.shp'))

print("\tJoining Wards to LSOAs...")

# Perform a spatial join between the LSOAs 2011 centroids and the ward_geo2011 GeoDataFrame
t_2011 = gpd.sjoin(lsoa_c2011, ward_geo2011, how='left')

# Rename columns in the resulting joined GeoDataFrame
t_2011.rename(columns={
    'GSS_CODE':'gss_cd',
    'LB_GSS_CD':'lb_gss_cd'
}, inplace=True)

# Save the relevant columns from the joined GeoDataFrame to a CSV file
t_2011[['lsoacd','gss_cd','lb_gss_cd']].to_csv(os.path.join(lkp,'LSOA_WARD_2011_JR.csv'), index=False)

print("Done.")

#%%
# Load the OAs 2011 path
oas_2011 = glob.glob(os.path.join(shpt, 'statistical-gis-boundaries-london', 'ESRI', 'OA_*.shp'))[0]

# Read the OAs 2011 shapefile into a GeoDataFrame
print("Processing Output Areas...")
oa_geo_2011 = gpd.read_file(oas_2011)
oa_geo_2011.crs = {'init': 'epsg:27700'}

# Save the OAs 2011 centroids as a new shapefile
print("\tSaving to shapefile...")
oa_geo_2011.to_file(os.path.join(shp, 'OAs 2011.shp'))

# Create a copy of the OAs 2011 GeoDataFrame
oa_c_2011 = oa_geo_2011
# Convert the geometry of the OAs to their centroids
oa_c_2011.geometry = oa_c_2011.centroid

# Save the OAs 2011 centroids as a new shapefile
print("\tSaving point OAs...")
oa_c_2011.to_file(os.path.join(shp, 'OAs 2011 Points.shp'))

print("\tOutput Areas to OAs...")

# Rename columns in the resulting joined GeoDataFrame
oa_geo_2011.rename(columns={
    'LSOA11CD': 'lsoacd',
    'OA11CD': 'oacd'
}, inplace=True)
# Save the relevant columns from the joined GeoDataFrame to a CSV file
oa_geo_2011[['lsoacd', 'oacd']].to_csv(os.path.join(lkp, 'LSOA_OA_2011_JR.csv'), index=False)

print("Done.")


# Rename columns in the resulting joined GeoDataFrame
oa_geo_2011.rename(columns={
    'LSOA11CD': 'lsoacd',
    'MSOA11CD': 'msoacd'
}, inplace=True)
# Save the relevant columns from the joined GeoDataFrame to a CSV file
oa_geo_2011[['lsoacd', 'msoacd']].to_csv(os.path.join(lkp, 'LSOA_MSOA_2011_JR.csv'), index=False)
msoas_loas_2011=oa_geo_2011[['lsoacd', 'msoacd']]
msoas_loas_2011.drop_duplicates(subset='lsoacd',inplace=True)
msoas_loas_2011.to_csv(os.path.join(lkp, 'LSOA_MSOA_2011_JR.csv'), index=False)
print("Done.")
#%%
print("Processing Coordinates...")
import pyproj
lsoa_g.set_index('LSOA11CD', inplace=True)
coords2011= lsoa_g.centroid.get_coordinates()
# Specify the original CRS details
original_crs = pyproj.CRS.from_epsg(27700)
# Replace <original_epsg_code> with the appropriate EPSG code of your original CRS.
# For example, if it's EPSG:4326 (WGS 84 - latitude and longitude in degrees), you can use: original_crs = pyproj.CRS.from_epsg(4326)

# Specify the target CRS details (WGS 84 - latitude and longitude in degrees)
target_crs = pyproj.CRS.from_epsg(4326)

# Create a Transformer to perform the coordinate conversion
transformer = pyproj.Transformer.from_crs(original_crs, target_crs, always_xy=True)

# Convert the 'x' and 'y' coordinates to degrees
coords2011['x_degrees'], coords2011['y_degrees'] = transformer.transform(coords2011['x'].values, coords2011['y'].values)

# Save the OAs 2011 centroids as a new shapefile
print("\tSaving to csv...")
coords2011.to_csv(os.path.join(shp, 'LSOAs Coordinates 2011.cvs'),index=True)
print("\tSaved")
#%% Local Authorities 2011
print("Processing: ")
la2011 = pd.read_csv(os.path.join(zip_folder,'nlac-2011.csv'))

# Rename columns for consistency
la2011.rename(columns={
    'New LA Code': 'LAD11CD',
    'LA Name': 'LAD11NM'
}, inplace=True)

# And select only those boroughs that match the array create above
la2011 = la2011[la2011['LAD11NM'].isin(boroughs)]
la2011[['LAD11CD', 'LAD11NM']].to_csv(os.path.join(lkp, 'LOCAL_AUTHORITIES_2011.csv'), index=False)


#%% 2021.
# Use glob to find the LSOA 2021 shapefile
lsoag2021 = glob.glob(os.path.join(shpt, '*LSOA_2021*.shp'))[0]

print("Processing: " + lsoag2021)

# Read the LSOA 2021 shapefile into a GeoDataFrame
lsoa_g2021 = gpd.read_file(lsoag2021)

# Extract borough/council names from LSOA names for subsetting
lsoa_g2021['borough'] = lsoa_g2021.LSOA21NM.str.extract("^(.+?) [0-9A-Z]{4}")

# Select only those LSOAs that belong to the specified boroughs
lsoa_g2021 = lsoa_g2021[lsoa_g2021.borough.isin(boroughs)]
lsoa21cd_to_drop = ['E01034224', 'E01034226']
lsoa_g2021 = lsoa_g2021[~lsoa_g2021['LSOA21CD'].isin(lsoa21cd_to_drop)]

print("\tExtracted " + str(len(lsoa_g2021)) + " London LSOAs.")

print("\tSaving to shapefile...")
lsoa_g2021.to_file(os.path.join(shp, 'LSOA-2021-Weights.shp'))

# Generate spatial weights using Queen contiguity from the shapefile
w = weights.Queen.from_shapefile(os.path.join(shp, 'LSOA-2021-Weights.shp'))

print("Check these results against above information:")
print("\tMean neighbors: " + str(w.mean_neighbors))
print("\tMax neighbors:  " + str(w.max_neighbors))
print("\tMin neighbors:  " + str(w.min_neighbors))
print("\tNo. Islands:    " + str(len(w.islands)))

# Set the coordinate reference system (CRS) of the LSOA 2021 GeoDataFrame
lsoa_g2021.crs = {'init': 'epsg:27700'}

# Insert a new column with a common name for LSOAs
lsoa_g2021.insert(0, 'lsoacd',lsoa_g2021[[x for x in lsoa_g2021.columns if 'LSOA' in x and ('CD' in x or 'CODE' in x)][0]])

print("\tSaving to shapefile...")
lsoa_g2021.to_file(os.path.join(shp, 'LSOAs 2021' + '.shp'))

print("\tSaving to pickle...")
lsoa_g2021.to_pickle(os.path.join(lkp, 'LSOAs 2021' + '.pkl'))

#%% #### Selecting and Joining Wards for 2021

wards2018 = glob.glob(os.path.join(shpt, 'London-wards-2018_ESRI', '*Ward*Merged.shp'))[0]

print("Processing wards...")
ward_geo2018 = gpd.read_file(wards2018)
ward_geo2018.crs = {'init': 'epsg:27700'}

# Rename the boroughs in the "DISTRICT" column
ward_geo2018['DISTRICT'] = ward_geo2018['DISTRICT'].replace(borough_rename)

print("\tSaving to shapefile...")
ward_geo2018.to_file(os.path.join(shp, 'Wards 2018.shp'))

#%% Create a mapping for LSOAs 2021 to Wards 2018
# Create a mapping for LSOAs to Wards
lsoa2021 = gpd.read_file(os.path.join(shp, 'LSOAs 2021.shp'))
lsoa2021.crs = {'init': 'epsg:27700'}

lsoa_c2021 = lsoa2021
lsoa_c2021.geometry = lsoa_c2021.centroid
lsoa_c2021.to_file(os.path.join(shp, 'LSOAs 2021 Points.shp'))

print("\tJoining Wards to LSOAs...")
t = gpd.sjoin(lsoa_c2021, ward_geo2018, how='left')
t.rename(columns={
    'GSS_CODE': 'gss_cd',
    'LAGSSCODE': 'lb_gss_cd',
    'LSOA21CD':'lsoacd'
}, inplace=True)

t[['lsoacd', 'gss_cd', 'lb_gss_cd']].to_csv(os.path.join(lkp, 'LSOA_WARD_2021_JR.csv'), index=False)

print("Done.")
#%%
print("\tOutput Areas to LSOAs...")
# Read the CSV file containing the mapping between Output Areas (OAs) and LSOAs
oas_loas_2021 = pd.read_csv(os.path.join(zip_folder, 'OA21_LSOA21_MSOA21_LAD22_EW_LU.csv'))

# Filter the mapping data to include only the specified boroughs
oas_loas_2021 = oas_loas_2021[oas_loas_2021['lad22nm'].isin(boroughs)]

# Rename columns for consistency
oas_loas_2021.rename(columns={
    'lsoa21cd': 'lsoacd',
    'oa21cd': 'oacd'
}, inplace=True)

# Save the LSOA-OA mapping to a CSV file
oas_loas_2021[['lsoacd', 'oacd']].to_csv(os.path.join(lkp, 'LSOA_OA_JR_2021.csv'), index=False)

print("Done.")

# Find the shapefile for Output Areas (OAs) 2021
oas2021 = glob.glob(os.path.join(shpt, 'OA_2021_EW_BGC.shp'))[0]

print("Processing Output Areas...")
# Read the OA shapefile into a GeoDataFrame
oa_geo2021 = gpd.read_file(oas2021)

# Set the coordinate reference system (CRS) of the OA GeoDataFrame
oa_geo2021.crs = {'init': 'epsg:27700'}

# Merge the OA GeoDataFrame with the LSOA-OA mapping DataFrame
merged_df = pd.merge(oa_geo2021, oas_loas_2021, left_on='OA21CD', right_on='oacd', how='inner')

print("\tSaving to shapefile...")
# Save the merged GeoDataFrame to a shapefile
merged_df.to_file(os.path.join(shp, 'OAs 2021.shp'))

# Create a new GeoDataFrame with OA centroids
oa_c2021 = merged_df
oa_c2021.geometry = oa_c2021.centroid

print("\tSaving point OAs...")
# Save the OA centroids GeoDataFrame to a shapefile
oa_c2021.to_file(os.path.join(shp, 'OAs 2021 Points.shp'))


#%%
print("\tMSOAs to LSOAs...")

# Read the CSV file containing the mapping between Output Areas (OAs) and LSOAs
msoas_loas_2021 = pd.read_csv(os.path.join(zip_folder, 'OA21_LSOA21_MSOA21_LAD22_EW_LU.csv'))

# Filter the mapping data to include only the specified boroughs
msoas_loas_2021 = msoas_loas_2021[msoas_loas_2021['lad22nm'].isin(boroughs)]

# Rename columns for consistency
msoas_loas_2021.rename(columns={
    'lsoa21cd': 'lsoacd',
    'msoa21cd': 'msoacd'
}, inplace=True)

# Save the LSOA-OA mapping to a CSV file
msoas_loas_2021=msoas_loas_2021[['lsoacd', 'msoacd']]
msoas_loas_2021.drop_duplicates(subset='lsoacd',inplace=True)
msoas_loas_2021.to_csv(os.path.join(lkp, 'LSOA_MSOA_JR_2021.csv'), index=False)

print("Done.")

#%% Local authorities
print("Processing: ")
la2021= pd.read_csv(os.path.join(zip_folder,'LAD_DEC_2021_UK_NC.csv'))

# And select only those boroughs that match the array create above
la2021 = la2021[la2021['LAD21NM'].isin(boroughs)]
la2021[['LAD21CD', 'LAD21NM']].to_csv(os.path.join(lkp, 'LOCAL_AUTHORITIES_2021.csv'), index=False)
print("Done")

#%%
shutil.rmtree(shpt)
print("Done.")