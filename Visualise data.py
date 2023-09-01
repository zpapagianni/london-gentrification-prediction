# Load libraries
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import geopandas as gpd
import os

warnings.filterwarnings("ignore")

# Define random state to reproduce
r_state = 58
random.seed(r_state)
np.random.seed(r_state)

from urllib.request import urlopen
import json
with urlopen('https://skgrange.github.io/www/data/london_boroughs.json') as response:
    boroughs = json.load(response)


# %%
# Create folders path
lkp = os.path.join('data', 'lkp')
shp = os.path.join('data', 'shp')
src = os.path.join('data', 'src')

analytical = os.path.join('data', 'analytical')
output = os.path.join('data', 'output')

# Create Directories
factors = os.path.join('data', 'factors')
housing = os.path.join(factors, 'housing')
demographics = os.path.join(factors, 'demographics')
work = os.path.join(factors, 'work')

boroughs = ['City of London', 'Barking and Dagenham', 'Barnet', 'Bexley', 'Brent', 'Bromley',
            'Camden', 'Croydon', 'Ealing', 'Enfield', 'Greenwich', 'Hackney', 'Hammersmith and Fulham',
            'Haringey', 'Harrow', 'Havering', 'Hillingdon', 'Hounslow', 'Islington',
            'Kensington and Chelsea', 'Kingston upon Thames', 'Lambeth', 'Lewisham',
            'Merton', 'Newham', 'Redbridge', 'Richmond upon Thames', 'Southwark', 'Sutton',
            'Tower Hamlets', 'Waltham Forest', 'Wandsworth', 'Westminster']

# Create a dictionary of borough names in 2018 and their corresponding 2011 names
borough_rename = {
    'City of Westminster': 'Westminster',
    'City and County of the City of London': 'City of London'}
# %%
df_11 = pd.read_csv(os.path.join(analytical, 'Untransformed-Data-2011.csv.gz'), index_col=0)
df_21 = pd.read_csv(os.path.join(analytical, 'Untransformed-Data-2021.csv.gz'), index_col=0)
score_diff = (df_21 / df_11 - 1) * 100

print('Loading shapefiles from 2011...')
lsoa_2011 = gpd.read_file(os.path.join(shp, 'LSOAs 2011.shp'))
ward_2011 = gpd.read_file(os.path.join(shp, 'Wards 2011.shp'))
mapping_data_2011 = pd.read_csv(os.path.join(lkp, 'LSOA_WARD_2011_JR.csv'))
print('Done')

# %%
df_11_mapped = pd.merge(mapping_data_2011, df_11, on='lsoacd', how='left')
df_21_mapped = pd.merge(mapping_data_2011, df_21, on='lsoacd', how='left')

df11_wards = df_11_mapped.groupby('gss_cd').sum().reset_index()
df11_wards = pd.merge(ward_2011, df11_wards, left_on='GSS_CODE', right_on='gss_cd', how='left')

df21_wards = df_21_mapped.groupby('gss_cd').sum().reset_index()
df21_wards = pd.merge(ward_2011, df21_wards, left_on='GSS_CODE', right_on='gss_cd', how='left')

diff_mapped = pd.merge(mapping_data_2011, score_diff, on='lsoacd', how='left')
diff_wards = diff_mapped.groupby('gss_cd').mean().reset_index()
diff_wards = pd.merge(ward_2011, diff_wards, left_on='GSS_CODE', right_on='gss_cd', how='left')

# %%
# Plot the London shapefile
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the London shapefile
lsoa_y.plot(ax=ax, alpha=0.5, edgecolor='k')
# Plot basemap
ctx.add_basemap(ax, crs=london.crs.to_string(), source=ctx.providers.CartoDB.Positron)
cx.add_basemap(ax, source=cx.providers.Stamen.TonerLabels)
ctx.add_basemap(ax,
                crs=diff_wards.crs.to_string(),
                source=ctx.providers.Stamen.Toner)
# Add a title and show the plot
plt.title('London LSOAs 2011')
plt.show()
# %%

# Compare population densities
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the first map
diff_wards.plot(
    column='House Prices',
    cmap='Spectral',
    linewidth=0.8,
    alpha=0.6,
    scheme="Natural_Breaks",
    k=8,
    ax=ax,  # Assign the first subplot
    edgecolor='0.8',
    legend=True,
    legend_kwds={"loc": 'lower right', "fmt": "{:.1f}"})

ctx.add_basemap(ax,
                crs=diff_wards.crs.to_string(),
                source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax,
                crs=diff_wards.crs.to_string(),
                source=ctx.providers.CartoDB.PositronOnlyLabels)

plt.title('Property Prices', fontsize=14)
# plt.legend(["% Diff"], fontsize=24)
# Hide axes
ax.axis('off')
plt.savefig('plots/property_map.png')
plt.show()

# %%
# Compare population densities
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the first map
diff_wards.plot(
    column='Median Income',
    cmap='Spectral',
    linewidth=0.8,
    alpha=0.6,
    scheme="Natural_Breaks",
    k=10,
    ax=ax,  # Assign the first subplot
    edgecolor='0.8',
    legend=True,
    legend_kwds={"loc": 'lower right', "fmt": "{:.1f}"})

ctx.add_basemap(ax,
                crs=df11_wards.crs.to_string(),
                source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax,
                crs=df11_wards.crs.to_string(),
                source=ctx.providers.CartoDB.PositronOnlyLabels)

plt.title('Median Income', fontsize=14)
# Hide axes
ax.axis('off')
plt.savefig('plots/income_map.png')
plt.show()

# %%
# Compare population densities
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the first map
diff_wards.plot(
    column='Share of Level 4+ Qualifications',
    cmap='Spectral',
    linewidth=0.8,
    alpha=0.6,
    scheme="Natural_Breaks",
    k=10,
    ax=ax,  # Assign the first subplot
    edgecolor='0.8',
    legend=True,
    legend_kwds={"loc": 'lower right', "fmt": "{:.1f}"})

ctx.add_basemap(ax,
                crs=df11_wards.crs.to_string(),
                source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax,
                crs=df11_wards.crs.to_string(),
                source=ctx.providers.CartoDB.PositronOnlyLabels)

plt.title('% Share of Level 4+ Qualifications', fontsize=14)
# Hide axes
ax.axis('off')
plt.savefig('plots/qual_map.png')
plt.show()

# %%
# Compare population densities
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the first map
diff_wards.plot(
    column='Share of Knowledge Workers',
    cmap='Spectral',
    linewidth=0.8,
    alpha=0.6,
    scheme="Natural_Breaks",
    k=10,
    ax=ax,  # Assign the first subplot
    edgecolor='0.8',
    legend=True,
    legend_kwds={"loc": 'lower right', "fmt": "{:.1f}"})

ctx.add_basemap(ax,
                crs=df11_wards.crs.to_string(),
                source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax,
                crs=df11_wards.crs.to_string(),
                source=ctx.providers.CartoDB.PositronOnlyLabels)

plt.title('% Share of Knowledge Workers', fontsize=14)
# Hide axes
ax.axis('off')
plt.savefig('plots/occup_map.png')
plt.show()

# %%
status = pd.read_csv(os.path.join(analytical, 'Scores.csv.gz'), index_col=0)  # score scores
status_mapped = pd.merge(mapping_data_2011, status, on='lsoacd', how='left')

geo_status = gpd.GeoDataFrame(status_mapped, geometry=lsoa_2011["geometry"])

# %%
# Compare population densities
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the first map

geo_status.plot(
    column='Score_ASC',
    cmap='Spectral',
    linewidth=0.8,
    alpha=0.6,
    scheme="Natural_Breaks",
    k=8,
    ax=ax,  # Assign the first subplot
    edgecolor='0.8',
    legend=True,
    legend_kwds={"loc": 'lower right', "fmt": "{:.1f}"})

ctx.add_basemap(ax,
                crs=london.crs.to_string(),
                source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax,
                crs=london.crs.to_string(),
                source=ctx.providers.CartoDB.PositronOnlyLabels)

plt.title('Gentrification Score Change (2011-2021)', fontsize=14)
# Hide axes
ax.axis('off')
plt.savefig('plots/gentrifications_score_2021.png')
# plt.savefig('plots/ascen_map.png')
plt.show()
# %%

rf_predicted = pd.read_csv(os.path.join(analytical, 'RF Predicted 2021.csv.gz'), index_col=0)  # score scores
rf_predicted = pd.merge(mapping_data_2011, rf_predicted, on='lsoacd', how='left')
geo_rf_predicted = gpd.GeoDataFrame(rf_predicted, geometry=lsoa_2011["geometry"])
london = gpd.read_file(os.path.join(shp, 'London.shp'))
# %%
# Compare population densities
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the first map

geo_rf_predicted.plot(
    column='Score Ascent 2011-2021 (Predicted)',
    cmap='Spectral',
    linewidth=0.8,
    alpha=0.6,
    scheme="Natural_Breaks",
    k=8,
    ax=ax,  # Assign the first subplot
    edgecolor='0.8',
    legend=True,
    legend_kwds={"loc": 'lower right', "fmt": "{:.1f}"})

ctx.add_basemap(ax,
                crs=london.crs.to_string(),
                source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax,
                crs=london.crs.to_string(),
                source=ctx.providers.CartoDB.PositronOnlyLabels, zoom=11)

plt.title('Gentrification Score Change Prediction (2011-2021) - Random Forest Model', fontsize=12)
# Hide axes
ax.axis('off')
plt.savefig('plots/rf-predicted.png')
plt.show()
# %%
gb_predicted = pd.read_csv(os.path.join(analytical, 'GB Predicted 2021.csv.gz'), index_col=0)  # score scores
gb_predicted = pd.merge(mapping_data_2011, gb_predicted, on='lsoacd', how='left')

geo_gb_predicted = gpd.GeoDataFrame(gb_predicted, geometry=lsoa_2011["geometry"])
# %%
# Compare population densities
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the first map

geo_gb_predicted.plot(
    column='Score Ascent 2011-2021 (Predicted)',
    cmap='Spectral',
    linewidth=0.8,
    alpha=0.6,
    scheme="Natural_Breaks",
    k=8,
    ax=ax,  # Assign the first subplot
    edgecolor='0.8',
    legend=True,
    legend_kwds={"loc": 'lower right', "fmt": "{:.1f}"})

ctx.add_basemap(ax,
                crs=london.crs.to_string(),
                source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax,
                crs=london.crs.to_string(),
                source=ctx.providers.CartoDB.PositronOnlyLabels, zoom=11)

plt.title('Gentrification Score Change Prediction (2011-2021) - Gradient Boosting Model', fontsize=12)
# Hide axes
ax.axis('off')
plt.savefig('plots/gb-predicted.png')
plt.show()

# %%
gb_predicted_2031 = pd.read_csv(os.path.join(analytical, 'GB-Full Predictions.csv.gz'), index_col=0)  # score scores
gb_predicted_2031 = pd.merge(mapping_data_2011, gb_predicted_2031, on='lsoacd', how='left')

geo_gb_predicted_2031 = gpd.GeoDataFrame(gb_predicted_2031, geometry=lsoa_2011["geometry"])
# %%
# Compare population densities
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the first map

geo_gb_predicted_2031.plot(
    column='Score Ascent 2021-2031 (Predicted)',
    cmap='Spectral',
    linewidth=0.8,
    alpha=0.6,
    scheme="Natural_Breaks",
    k=5,
    ax=ax,  # Assign the first subplot
    edgecolor='0.8',
    legend=True,
    legend_kwds={"loc": 'lower right', "fmt": "{:.1f}"})

ctx.add_basemap(ax,
                crs=london.crs.to_string(),
                source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax,
                crs=london.crs.to_string(),
                source=ctx.providers.CartoDB.PositronOnlyLabels)

plt.title('Gentrification Score Change Prediction (2021 to 2031) - Gradient Boosting Model', fontsize=14)
# Hide axes
ax.axis('off')
plt.savefig('plots/2031-predicted-gb.png')
plt.show()

# %%
rf_predicted_2031 = pd.read_csv(os.path.join(analytical, 'RF-Full Predictions.csv.gz'), index_col=0)  # score Scores
rf_predicted_2031 = pd.merge(mapping_data_2011, rf_predicted_2031, on='lsoacd', how='left')

geo_rf_predicted_2031 = gpd.GeoDataFrame(gb_predicted_2031, geometry=lsoa_2011["geometry"])
# %%
# Compare population densities
fig, ax = plt.subplots(figsize=(12, 12))
# Plot the first map

geo_rf_predicted_2031.plot(
    column='Score Ascent 2021-2031 (Predicted)',
    cmap='Spectral',
    linewidth=0.8,
    alpha=0.6,
    scheme="Natural_Breaks",
    k=5,
    ax=ax,  # Assign the first subplot
    edgecolor='0.8',
    legend=True,
    legend_kwds={"loc": 'lower right', "fmt": "{:.1f}"})

ctx.add_basemap(ax,
                crs=london.crs.to_string(),
                source=ctx.providers.CartoDB.Positron)
ctx.add_basemap(ax,
                crs=london.crs.to_string(),
                source=ctx.providers.CartoDB.PositronOnlyLabels, zoom=11)

plt.title('Gentrification Score Change Prediction (2021 to 2031) - Random Forest Model', fontsize=14)
# Hide axes
ax.axis('off')
plt.savefig('plots/2031-predicted-rf.png')
plt.show()
# %%
