# Load libraries
import os
import random
import re
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysal as ps
import seaborn as sns
from pysal.lib import weights
from scipy.stats import gmean

warnings.filterwarnings("ignore")

# Define random state to reproduce
r_state = 58
random.seed(r_state)
np.random.seed(r_state)

## Specify Directories
predictors = os.path.join('data', 'input', 'predictors')

folder2011 = os.path.join(predictors, '2011')
folder2021 = os.path.join(predictors, '2021')
combined = os.path.join(predictors, 'combined')

# Create Output Directories
factors = os.path.join('data', 'factors')
# Create folders path
lkp = os.path.join('data', 'lkp')
shp = os.path.join('data', 'shp')
src = os.path.join('data', 'src')
zip_folder = os.path.join('data', 'input', 'zip')

# Delete if it already exists
if os.path.exists(factors):
    shutil.rmtree(factors)

housing = os.path.join(factors, 'housing')
demographics = os.path.join(factors, 'demographics')
work = os.path.join(factors, 'work')
social = os.path.join(factors, 'social')
for d in [factors, demographics, housing, work, social]:
    if not os.path.exists(d):
        os.makedirs(d)

# Read the LSOAs data for 2011 and 2021
ldn2011 = pd.read_pickle(os.path.join(lkp, 'LSOAs 2011.pkl'))
ldn2021 = pd.read_pickle(os.path.join(lkp, 'LSOAs 2021.pkl'))

print("Have built London LSOA filter data for use where needed...")
print("\t2011: " + str(ldn2011.shape[0]) + " rows.")
print("\t2021: " + str(ldn2021.shape[0]) + " rows.")

# Define boroughs
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
# Functions
def get_neighbours(ns, fh):
    """
    Find neighbours of a given LSOA.
    """
    neighbours = []
    for n in ns.keys():
        neighbours.append(fh[n][0][0])
    return neighbours


def get_gmean_from_neighbours(ns, prices):
    """
    Find geometric mean of an LSOAs _neighbours'_ property transactions.
    """
    # Searching for areas
    medians = prices.loc[prices.index.isin(ns), 'Median Property Price'].values
    # Find median prices
    return round(gmean(medians[np.logical_not(np.isnan(medians))]), -1)


def check_missing_values(df):
    nan_count = df.isna().sum().sum()
    if nan_count == 0:
        print("No missing values")
    else:
        print(f"The DataFrame contains {nan_count} NaN values.")
    return


# Function for sanity check
def check_dataframe_columns(df1, df2):
    if df1.shape[1] != df2.shape[1]:
        print("Number of columns is different.")
        print("Columns in the first DataFrame:", df1.shape[1])
        print("Columns in the second DataFrame:", df2.shape[1])
    else:
        print('Same number of columns')

    if len(df1.columns.difference(df2.columns)) == 0:
        print("Both DataFrames have the same columns.")
    else:
        print('Columns Only Contained in 2011:')
        print(df1.columns.difference(df2.columns))

    if len(df2.columns.difference(df1.columns)) == 0:
        print("Both DataFrames have the same columns.")
    else:
        print('Columns Only Contained in 2021')
        print(df2.columns.difference(df1.columns))
    return


def fill_missing_values_2(df, qw2011, fh2011, cds2011, max_iterations=5):
    nan_rows = df[df.isnull().any(axis=1)]['lsoacd'].values

    print(f"Looking for neighbors of {len(nan_rows)} areas without values.")
    print("Filling missing values...")
    iterations = 0
    missing_values_exist = 1

    while missing_values_exist != 0 and iterations < max_iterations:
        print(f'Iteration {iterations}')
        print(f'Number of missing values {missing_values_exist}')
        missing_values_exist = 0

        for column in df.columns[1:]:
            print(f"Processing column: {column}")
            # Iterate over rows with missing values
            for row_index in nan_rows:
                # Find the neighbors of the current LSOA
                neighbors = get_neighbours(qw2011[cds2011.index(row_index)], fh2011)
                # Calculate the mean of the neighbors' variables
                variables = df.loc[df['lsoacd'].isin(neighbors)]
                # Calculate the mean of each variable across the neighbors (ignoring NaN values)
                means = variables.mean(axis=0, skipna=True)
                # Assign the mean value to the current row and column if it is not NaN
                if not np.isnan(means.values).any():
                    df.loc[df['lsoacd'] == row_index, 1:] = means.values
                else:
                    missing_values_exist += 1

        iterations += 1

    print("Done")

    return df


# %%
lsoas2021_data = pd.read_csv(os.path.join(lkp, 'LSOA_WARD_2021_JR.csv'))
lsoas2011_data = pd.read_csv(os.path.join(lkp, 'LSOA_WARD_2011_JR.csv'))
lookup_table = pd.read_csv(os.path.join(zip_folder, 'LSOA_(2011)_to_LSOA_(2021).csv'))


# Function to convert LSOA11 codes to LSOA21 codes and process the data
def convert_lsoa21_to_lsoa11(df, lookup_data, lsoas2011):
    # Choose London boroughs by filtering the lookup_data DataFrame based on the 'NAME' column
    lookup_data = lookup_data.loc[lookup_data.LSOA11CD.isin(lsoas2011.lsoacd)].reset_index(drop=True)
    # Filter out rows where chgind is equal to 'X'
    lookup_data = lookup_data[lookup_data['CHGIND'] != 'X']

    # Merge the input DataFrame with the lookup_data DataFrame based on LSOA codes
    merged_df = pd.merge(df, lookup_data, left_on='lsoacd', right_on='LSOA21CD', how='right')

    # Define column names for the processed DataFrame
    geo_names = ['LSOA11CD', 'LSOA21CD', 'CHGIND']
    column_names = list(df.columns)
    column_names.remove('lsoacd')
    geo_names.extend(column_names)

    # Select and reorder columns for the processed DataFrame
    processed_df = merged_df[geo_names]
    processed_rows = []
    processed2011lsoas = []
    processed2021lsoas = []

    final_df = processed_df[processed_df['CHGIND'] == 'U']
    final_df.drop(['LSOA21CD', 'CHGIND'], axis=1, inplace=True)

    # Iterate through each row of the processed DataFrame
    for _, row in processed_df.iterrows():
        if row['CHGIND'] == 'M':
            if row['LSOA21CD'] not in processed2021lsoas:
                processed2021lsoas.append(row['LSOA21CD'])
                # Get the corresponding aggregated 2011 LSOAs that were merged
                merged_lsoa11cd_rows = processed_df.loc[processed_df['LSOA21CD'] == row['LSOA21CD']]
                allocated_data = merged_lsoa11cd_rows.iloc[0, 3:] / len(merged_lsoa11cd_rows)
                for _, s_row in merged_lsoa11cd_rows.iterrows():
                    processed_rows.append(pd.concat([pd.Series(s_row['LSOA11CD']), allocated_data]))
        elif row['CHGIND'] == 'S':
            if row['LSOA11CD'] not in processed2011lsoas:
                same_lsoa11cd_rows = processed_df.loc[processed_df['LSOA11CD'] == row['LSOA11CD']]
                aggregated_data = same_lsoa11cd_rows.sum(numeric_only=True)  # Aggregate the data (e.g., using sum)
                processed_rows.append(pd.concat([pd.Series(row['LSOA11CD']), aggregated_data]))
                processed2011lsoas.append(row['LSOA11CD'])

    len(processed_rows)

    # Create a new DataFrame with the processed rows and drop duplicate rows
    df_converted = pd.DataFrame(processed_rows)
    df_converted.rename(columns={0: 'LSOA11CD'}, inplace=True)
    df_converted = df_converted.drop_duplicates(subset=['LSOA11CD'])

    final_df1 = pd.concat([df_converted, final_df])
    final_df1.rename(columns={'LSOA11CD': 'lsoacd'}, inplace=True)

    final_df1.hist(bins=15)
    plt.show()

    return final_df1


# Function to convert LSOA11 codes to LSOA21 codes using mean.
def mean_convert_lsoa21_to_lsoa11(df, lookup_data, lsoas2011):
    # Choose London boroughs by filtering the lookup_data DataFrame based on the 'NAME' column
    lookup_data = lookup_data.loc[lookup_data.LSOA11CD.isin(lsoas2011.lsoacd)].reset_index(drop=True)
    # Filter out rows where chgind is equal to 'X'
    lookup_data = lookup_data[lookup_data['CHGIND'] != 'X']

    # Merge the input DataFrame with the lookup_data DataFrame based on LSOA codes
    merged_df = pd.merge(df, lookup_data, left_on='lsoacd', right_on='LSOA21CD', how='right')

    # Define column names for the processed DataFrame
    geo_names = ['LSOA11CD', 'LSOA21CD', 'CHGIND']
    column_names = list(df.columns)
    column_names.remove('lsoacd')
    geo_names.extend(column_names)

    # Select and reorder columns for the processed DataFrame
    processed_df = merged_df[geo_names]
    processed_rows = []
    processed2011lsoas = []
    processed2021lsoas = []

    final_df = processed_df[processed_df['CHGIND'] == 'U']
    final_df.drop(['LSOA21CD', 'CHGIND'], axis=1, inplace=True)

    # Iterate through each row of the processed DataFrame
    for _, row in processed_df.iterrows():
        if row['CHGIND'] == 'M':
            if row['LSOA21CD'] not in processed2021lsoas:
                processed2021lsoas.append(row['LSOA21CD'])
                # Get the corresponding aggregated 2011 LSOAs that were merged
                merged_lsoa11cd_rows = processed_df.loc[processed_df['LSOA21CD'] == row['LSOA21CD']]
                allocated_data = merged_lsoa11cd_rows.median(numeric_only=True)
                for _, s_row in merged_lsoa11cd_rows.iterrows():
                    processed_rows.append(pd.concat([pd.Series(s_row['LSOA11CD']), allocated_data]))
        elif row['CHGIND'] == 'S':
            if row['LSOA11CD'] not in processed2011lsoas:
                same_lsoa11cd_rows = processed_df.loc[processed_df['LSOA11CD'] == row['LSOA11CD']]
                aggregated_data = same_lsoa11cd_rows.median(numeric_only=True)  # Aggregate the data (e.g., using sum)
                processed_rows.append(pd.concat([pd.Series(row['LSOA11CD']), aggregated_data]))
                processed2011lsoas.append(row['LSOA11CD'])

    len(processed_rows)

    # Create a new DataFrame with the processed rows and drop duplicate rows
    df_converted = pd.DataFrame(processed_rows)
    df_converted.rename(columns={0: 'LSOA11CD'}, inplace=True)
    df_converted = df_converted.drop_duplicates(subset=['LSOA11CD'])

    final_df1 = pd.concat([df_converted, final_df])
    final_df1.rename(columns={'LSOA11CD': 'lsoacd'}, inplace=True)

    final_df1.hist(bins=15)
    plt.show()

    return final_df1


# %%
# Median House Price
median_hp = pd.read_csv(os.path.join(combined, 'Median-House-Prices-1995-2022.csv'))

# Simplify column names
median_hp.rename(columns={
    'LSOA code': 'lsoacd',
    'LSOA name': 'name',
    'Local authority name': 'borough'}
    , inplace=True)

median_hp = median_hp[median_hp['borough'].isin(boroughs)]
median_hp.set_index('lsoacd', inplace=True)

# And break them down into subsets
mhp2011 = median_hp.loc[:, ['Dec-11']]
mhp2021 = median_hp.loc[:, ['Dec-21']]

# Rename keys for consistency
mhp2011.rename(columns={'Dec-11': 'Median Property Price'}, inplace=True)
mhp2021.rename(columns={'Dec-21': 'Median Property Price'}, inplace=True)

# Sanity check
print("Have " + str(mhp2011.shape[0]) + " rows of data.")
print("Done.")

# Check missing values
check_missing_values(mhp2011)
check_missing_values(mhp2021)

plt.figure(figsize=(10, 6))
sns.heatmap(mhp2011.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
plt.show()

# %%
# Define the weights and adjacency for 2011
qw2011 = weights.Queen.from_shapefile(os.path.join('data', 'shp', 'LSOA-2011-Weights.shp'))
fh2011 = ps.lib.io.open(os.path.join('data', 'shp', 'LSOA-2011-Weights.dbf'))
cds2011 = fh2011.by_col['LSOA11CD']  # LSOA 2011 Census code

print("2011...")
nan11 = mhp2011[mhp2011['Median Property Price'].isnull()].index.values
print("\tLooking for neighbours of " + str(len(nan11)) + " areas without house prices.")

# Iterate over areas without house prices in 2011
for z in nan11:
    neighbours11 = get_neighbours(qw2011[cds2011.index(z)], fh2011)
    m = get_gmean_from_neighbours(neighbours11, mhp2011)
    mhp2011.loc[z, 'Median Property Price'] = m

print("Done ")

# Repeat for 2021
print("2021...")
nan21 = mhp2021[mhp2021['Median Property Price'].isnull()].index.values
print("\tLooking for neighbours of " + str(len(nan21)) + " areas without house prices.")

# Iterate over areas without house prices in 2021
for z in nan21:
    neighbours21 = get_neighbours(qw2011[cds2011.index(z)], fh2011)
    m = get_gmean_from_neighbours(neighbours21, mhp2021)
    mhp2021.loc[z, 'Median Property Price'] = m

print(" ")

# Save the data to CSV files
mhp2011.loc[:, ['Median Property Price']].to_csv(os.path.join(housing, 'Property-Price-2011.csv'), index=True,
                                                 header=True, encoding='utf-8')
mhp2021.loc[:, ['Median Property Price']].to_csv(os.path.join(housing, 'Property-Price-2021.csv'), index=True,
                                                 header=True, encoding='utf-8')

print("Saved.")

mhp2011.sample(3, random_state=r_state)

# %%
sales_num = pd.read_csv(os.path.join(combined, 'Numberofresidentialpropertysalesbylsoa.csv'))
sales_num.rename(columns={
    'LSOA code': 'lsoacd',
    'LSOA name': 'name'}
    , inplace=True)

# Drop the non-London LSOAs
sales_num.drop(['name'], axis=1, inplace=True)
sales_num = sales_num[sales_num.lsoacd.isin(ldn2011.lsoacd.values)]

sales_num.rename(columns=lambda x: re.sub('Year ending ', '', x), inplace=True)
sales_num.set_index('lsoacd', inplace=True)
# And break them down into subsets
sales_num2011 = sales_num.loc[:, ['Dec 2011']]
sales_num2021 = sales_num.loc[:, ['Dec 2021']]

# Rename keys for consistency
sales_num2011.rename(columns={'Dec 2011': 'Number of Sales'}, inplace=True)
sales_num2021.rename(columns={'Dec 2021': 'Number of Sales'}, inplace=True)

# Sanity check
print("Have " + str(sales_num2011.shape[0]) + " rows of data.")
print("Done.")

# Check missing values
sales_num2011.to_csv(os.path.join(housing, 'Number-of-Sales-2011.csv'), index=True, header=True, encoding='utf-8')
sales_num2021.to_csv(os.path.join(housing, 'Number-of-Sales-2021.csv'), index=True, header=True, encoding='utf-8')

# check_plots(sales_num2011, sales_num2011)


# %%
dwelling_period = pd.read_csv(os.path.join(combined, 'dwelling-period-built-2014-lsoa.csv'))
dwelling_period.rename(columns={
    'lsoa': 'lsoacd',
    'Name': 'name',
    'UNKNOWN': 'Other'}
    , inplace=True)

# Drop the non-London LSOAs
dwelling_period.drop(['GEOG', 'name'], axis=1, inplace=True)
dwelling_period = dwelling_period[dwelling_period.lsoacd.isin(ldn2011.lsoacd.values)]
dwelling_period.rename(columns=lambda x: re.sub('_', '-', x), inplace=True)

# Sanity check
print("Have " + str(dwelling_period.shape[0]) + " rows of data.")

# Check missing values
dwelling_period.fillna(0, inplace=True)
dwelling_period.sample(3, random_state=r_state)

period_agg = pd.DataFrame()
period_agg['lsoacd'] = dwelling_period['lsoacd']
period_agg['Pre-1929'] = dwelling_period['Pre-1900'] + dwelling_period['1900-1918'] + dwelling_period['1919-1929']
period_agg['1930-1964'] = dwelling_period['1930-1939'] + dwelling_period['1945-1954'] + dwelling_period['1955-1964']
period_agg['1965-1992'] = dwelling_period['1965-1972'] + dwelling_period['1973-1982'] + dwelling_period['1983-1992']
period_agg['1993-1999'] = dwelling_period['1993-1999']
period_agg['2000-2009'] = dwelling_period['2000-2009']
period_agg['2010-2014'] = dwelling_period['2010-2014']
period_agg.sample(3, random_state=r_state)

period_agg.hist()
plt.show()

period_agg.to_csv(os.path.join(housing, 'Dwelling-Period.csv'), index=False, header=True, encoding='utf-8')

# %%Accommodation Type

print("Processing Accommodation Type files ...")
acctype2011 = pd.read_csv(os.path.join(folder2011, 'census2011-accomodation-lsoa.csv'))

# Convert the columns names
acctype2011.rename(columns={
    'mnemonic': 'lsoacd',
    '2011 super output area - lower layer': 'lsoanm',
    'All categories: Accommodation type': 'total'
}, inplace=True)
acctype2011.rename(
    columns=lambda x: re.sub('^Unshared dwelling(?:\: Whole house or bungalow|\: Flat, maisonette or apartment)?: ', '',
                             x), inplace=True)
acctype2011.rename(columns=lambda x: re.sub(' \(.+?\)', '', x), inplace=True)

# Drop the non-London LSOAs
acctype2011.drop(['lsoanm'], axis=1, inplace=True)
acctype2011 = acctype2011[acctype2011.lsoacd.isin(ldn2011.lsoacd.values)]

acctype2011.sample(3, random_state=r_state)

# Repeat process for 2021
acctype2021 = pd.read_csv(os.path.join(folder2021, 'census2021-accomodation-lsoa.csv'))
acctype2021.drop(['date'], axis=1, inplace=True)
# Convert the column names to something more tractable
acctype2021.rename(columns={
    'geography code': 'lsoacd',
    'geography': 'lsoanm',
    'Accommodation type: Total: All households': 'total'
}, inplace=True)

acctype2021.rename(columns=lambda x: re.sub('Accommodation type: ', '', x), inplace=True)
acctype2021.rename(columns=lambda x: re.sub(' \(.+?\)', '', x), inplace=True)

acctype2021['Part of a converted or shared house'] = acctype2021[
                                                         'Part of a converted or shared house, including bedsits'] + \
                                                     acctype2021[
                                                         'Part of another converted building, for example, former school, church or warehouse']
acctype2021.drop(['Part of a converted or shared house, including bedsits',
                  'Part of another converted building, for example, former school, church or warehouse'], axis=1,
                 inplace=True)

acctype2021.rename(columns={
    'In a purpose-built block of flats or tenement': 'Purpose-built block of flats or tenement',
    'In a commercial building, for example, in an office building, hotel or over a shop': 'In commercial building',
    'A caravan or other mobile or temporary structure': 'Caravan or other mobile or temporary structure'
}, inplace=True)

# Drop the non-London LSOAs
acctype2021.drop(['lsoanm'], axis=1, inplace=True)
acctype2021 = acctype2021[acctype2021.lsoacd.isin(ldn2021.lsoacd.values)]

# Sanity check
print("2021 Accommodation Type data frame contains " + str(acctype2021.shape[0]) + " rows.")
acctype2021.sample(3, random_state=r_state)

check_dataframe_columns(acctype2011, acctype2021)

print('Saving...')
# Save all
acctype2011.to_csv(os.path.join(housing, 'Accommodation Type-2011.csv'), index=False)
acctype2021.to_csv(os.path.join(housing, 'Accommodation Type-2021.csv'), index=False)
acctype2021_conv = convert_lsoa21_to_lsoa11(acctype2021, lookup_table, lsoas2011_data)
acctype2021_conv.to_csv(os.path.join(housing, 'Accommodation Type-Converted-2021.csv'), index=False)
print('Done')

# %% Tenure
print("Processing tenure files ...")
ten2011 = pd.read_csv(os.path.join(folder2011, 'census2011-tenure-lsoa.csv'))

# Rename columns
ten2011.rename(columns={
    '2011 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'All households': 'total',
    'Owned: Owned outright': 'Owned: Outright',
    'Owned: Owned with a mortgage or loan': 'Owned: Mortgaged',
    'Shared ownership (part owned and part rented)': 'Shared ownership',
    'Social rented: Rented from council (Local Authority)': 'Rented: Council',
    'Social rented: Other': 'Rented: HA or RSL',
    'Private rented: Private landlord or letting agency': 'Rented: Private'
}, inplace=True)

# Drop
ten2011.drop(['lsoanm'], axis=1, inplace=True)
ten2011.drop(['Owned', 'Social rented', 'Private rented'], axis=1, inplace=True)
# Choose london boroughs
ten2011 = ten2011[ten2011.lsoacd.isin(ldn2011.lsoacd.values)]
# Check
ten2011.sample(3, random_state=r_state)

# 2021
ten2021 = pd.read_csv(os.path.join(folder2021, 'census2021-tenure-lsoa.csv'))

ten2021.rename(columns=lambda x: re.sub('Tenure of household: ', '', x), inplace=True)
# Shared ownership: Shared ownership is duplicate of Shared ownership
ten2021.drop(['date', 'Shared ownership: Shared ownership'], axis=1, inplace=True)

# Macth column names with 2011
ten2021.rename(columns={
    'geography': 'lsoanm',
    'geography code': 'lsoacd',
    'Total: All households': 'total',
    'Owned: Owns outright': 'Owned: Outright',
    'Owned: Owns with a mortgage or loan': 'Owned: Mortgaged',
    'Social rented: Rents from council or Local Authority': 'Rented: Council',
    'Social rented: Other social rented': 'Rented: HA or RSL',
    'Private rented: Private landlord or letting agency': 'Rented: Private',
    'Private rented: Other private rented': 'Private rented: Other',
    'Lives rent free': 'Living rent free'
}, inplace=True)

# Drop unnecessary columns
ten2021.drop(['lsoanm'], axis=1, inplace=True)
ten2021.drop(['Owned', 'Social rented', 'Private rented'], axis=1, inplace=True)
ten2021 = ten2021[ten2021.lsoacd.isin(ldn2021.lsoacd.values)]

# Sanity check, should be 4994
print("Have " + str(ten2021.shape[0]) + " rows of data.")
ten2021.sample(3, random_state=r_state)

# Column names check
check_dataframe_columns(ten2011, ten2011)

print('Saving...')
# Save all
ten2011.to_csv(os.path.join(housing, 'Tenure-2011.csv'), index=False)
ten2021.to_csv(os.path.join(housing, 'Tenure-2021.csv'), index=False)
ten2021_conv = convert_lsoa21_to_lsoa11(ten2021, lookup_table, lsoas2011_data)
ten2021_conv.to_csv(os.path.join(housing, 'Tenure-Converted-2021.csv'), index=False)

print('Saved')

# %%
print("Processing private rents files ...")
rents_2011 = pd.read_excel(os.path.join(folder2011, 'Rents 2011.xls'), skiprows=6,
                           sheet_name='Table2.7')
# Drop columns
rents_2011 = rents_2011[['Area Code2', 'Area', 'Median']]

# Rename columns
rents_2011.rename(columns={
    'Area Code2': 'ladcd',
    'Area': 'ladnm',
    'Median': 'Median Rent'
}, inplace=True)

# Choose london boroughs
la2011 = pd.read_csv(os.path.join(lkp, 'LOCAL_AUTHORITIES_2011.csv'))
rents_2011 = rents_2011[rents_2011.ladcd.isin(la2011.LAD11CD.values)]

# Sanity check, should be 33
print("Have " + str(rents_2011.shape[0]) + " rows of data.")
rents_2011.sample(3, random_state=r_state)

# 2021
rents_2021 = pd.read_excel(os.path.join(folder2021, 'Rents-2021.xlsx'), skiprows=6,
                           sheet_name='Table2.7')

# Drop columns
rents_2021 = rents_2021[['Area Code1', 'Area', 'Median']]

# Rename columns
rents_2021.rename(columns={
    'Area Code1': 'ladcd',
    'Area': 'ladnm',
    'Median': 'Median Rent'
}, inplace=True)

# Choose london boroughs
la2021 = pd.read_csv(os.path.join(lkp, 'LOCAL_AUTHORITIES_2021.csv'))
rents_2021 = rents_2021[rents_2021.ladcd.isin(la2021.LAD21CD.values)]

# Sanity check, should be 33
print("Have " + str(rents_2021.shape[0]) + " rows of data.")
rents_2021.sample(3, random_state=r_state)

print('Saving...')
# Save all
rents_2011.to_csv(os.path.join(housing, 'Private-rents-2011.csv'), index=False)
rents_2021.to_csv(os.path.join(housing, 'Private-rents-2021.csv'), index=False)
print('Saved')

# %%
# Read borough data for the year 2011 from a CSV file
borough_data_2011 = pd.read_csv(os.path.join(zip_folder, 'PCD11_OA11_LSOA11_MSOA11_LAD11_EW_LU_aligned_v2.csv'))

# Filter borough data to include only the specified boroughs and reset the index
borough_data_2011 = borough_data_2011.loc[borough_data_2011.LAD11NM.isin(boroughs)].reset_index(drop=True)

# Select and retain only the necessary columns from the borough data
borough_data_2011 = borough_data_2011[['LAD11CD', 'LSOA11CD']]

# Remove duplicate entries from the borough data
borough_data_2011.drop_duplicates(inplace=True)

# Rename columns for consistency
borough_data_2011.rename(columns={'LAD11CD': 'ladcd',
                                  'LSOA11CD': 'lsoacd'}, inplace=True)

# Filter lookup table for the year 2021 to include only relevant LSOAs and reset the index
borough_data_2021 = lookup_table.loc[lookup_table.LSOA21CD.isin(lsoas2021_data['lsoacd'])].reset_index(drop=True)

# Select and retain only necessary columns from the filtered lookup table
borough_data_2021 = borough_data_2021[['LAD22CD', 'LSOA21CD']]

# Remove duplicate entries from the borough data for 2021
borough_data_2021.drop_duplicates(inplace=True)

# Rename columns for consistency
borough_data_2021.rename(columns={'LAD22CD': 'ladcd',
                                  'LSOA21CD': 'lsoacd'}, inplace=True)

# Merge borough data for 2011 with rent data for 2011 based on local authority code (ladcd)
rents_2011_lsoas = pd.merge(borough_data_2011, rents_2011, on='ladcd', how='left')

# Drop unnecessary columns from the merged data
rents_2011_lsoas.drop(['ladcd', 'ladnm'], axis=1, inplace=True)

# Save the merged and processed data to a CSV file for 2011
rents_2011_lsoas.to_csv(os.path.join(housing, 'Private-rents-LSOAs-2011.csv'), index=False)

# Merge borough data for 2021 with rent data for 2021 based on local authority code (ladcd)
rents_2021_lsoas = pd.merge(borough_data_2021, rents_2021, on='ladcd', how='left')

# Drop unnecessary columns from the merged data
rents_2021_lsoas.drop(['ladcd', 'ladnm'], axis=1, inplace=True)

# Convert LSOA codes from 2021 to 2011 using a conversion function
rents_2021_lsoas_conv = mean_convert_lsoa21_to_lsoa11(rents_2021_lsoas, lookup_table, lsoas2011_data)

# Save the converted and processed data to a CSV file for 2021
rents_2021_lsoas_conv.to_csv(os.path.join(housing, 'Private-rents-LSOAs-Converted-2021.csv'), index=False)

# %%
print("Processing household income ...")
hi_2011 = pd.read_csv(os.path.join(folder2011, 'income-2011.csv'), skiprows=5)

# Drop columns
hi_2011 = hi_2011[['County and district / unitary authority', 'Total income-Median']]

# Rename columns
hi_2011.rename(columns={
    'County and district / unitary authority': 'ladnm',
    'Total income-Median': 'Income'
}, inplace=True)

hi_2011['ladnm'] = hi_2011['ladnm'].replace({'Kingston-upon-Thames': 'Kingston upon Thames',
                                             'Richmond-upon-Thames': 'Richmond upon Thames'})

# Choose london boroughs
hi_2011 = hi_2011[hi_2011.ladnm.isin(boroughs)]

# Merge the OA GeoDataFrame with the LSOA-OA mapping DataFrame
hi_2011 = pd.merge(hi_2011, la2011, left_on='ladnm', right_on='LAD11NM', how='inner')

hi_2011 = hi_2011[['LAD11CD', 'ladnm', 'Income']]

hi_2011.rename(columns={
    'LAD11CD': 'ladcd'}, inplace=True)

# Sanity check, should be 33
print("Have " + str(hi_2011.shape[0]) + " rows of data.")
hi_2011.sample(3, random_state=r_state)

# 2021
hi_2021 = pd.read_csv(os.path.join(folder2021, 'income-2021.csv'), skiprows=4)

# Rename columns
hi_2021.rename(columns={
    'Area Code': 'ladcd',
    'Area Name': 'ladnm',
    'Total income: Median': 'Income'
}, inplace=True)

# Drop columns
hi_2021 = hi_2021[['ladcd', 'ladnm', 'Income']]

# Choose london boroughs
hi_2021 = hi_2021[hi_2021.ladcd.isin(la2021.LAD21CD.values)]

# Sanity check, should be 33
print("Have " + str(hi_2021.shape[0]) + " rows of data.")
hi_2021.sample(3, random_state=r_state)

print('Saving...')
# Save all
hi_2011.to_csv(os.path.join(housing, 'Private-Income-2011.csv'), index=False)
hi_2021.to_csv(os.path.join(housing, 'Private-Income-2021.csv'), index=False)

print('Saved')

# check_plots(hi_2011, hi_2021)

# %%
hi_2011_lsoas = pd.merge(borough_data_2011, hi_2011, on='ladcd', how='left')
hi_2011_lsoas.drop(['ladcd', 'ladnm'], axis=1, inplace=True)
hi_2011_lsoas.to_csv(os.path.join(housing, 'Private-Income-LSOAs-2011.csv'), index=False)

hi_2021_lsoas = pd.merge(borough_data_2021, hi_2021, on='ladcd', how='left')
hi_2021_lsoas.drop(['ladcd', 'ladnm'], axis=1, inplace=True)
hi_2021_lsoas_conv = mean_convert_lsoa21_to_lsoa11(hi_2021_lsoas, lookup_table, lsoas2011_data)
hi_2021_lsoas_conv.to_csv(os.path.join(housing, 'Private-Income-LSOAs-Converted-2021.csv'), index=False)

# %% Income MSOA
inc2011 = pd.read_csv(os.path.join(folder2011, 'income-msoas-2011.csv'))

inc2011.rename(columns={
    'MSOA name': 'msoanm',
    'MSOA code': 'msoacd',
    'Total weekly income': 'Weekly income',
}, inplace=True)

# Add on LSOA data
lsoa2msoa = pd.read_csv(os.path.join(lkp, 'LSOA_MSOA_2011_JR.csv'))
inc2011_lsoa = pd.merge(lsoa2msoa, inc2011, left_on='msoacd', right_on='msoacd', how='inner')
inc2011_lsoa['Income'] = inc2011_lsoa['Weekly income'] * 52

# Tidy up
inc2011_lsoa.drop(['msoacd', 'msoanm', 'Local authority code',
                   'Local authority name', 'Region code', 'Region name', 'Weekly income',
                   'Upper confidence limit', 'Lower confidence limit',
                   'Confidence interval'], axis=1, inplace=True)

# Sanity check
print("Have " + str(inc2011_lsoa.shape[0]) + " rows data.")
print("Done.")
inc2011_lsoa.sample(3, random_state=r_state)

inc2021 = pd.read_csv(os.path.join(folder2021, 'income-msoas-2021.csv'))
inc2021.rename(columns={
    'MSOA name': 'msoanm',
    'MSOA code': 'msoacd',
    'Total annual income': 'Income',
}, inplace=True)

# Add on LSOA data
inc2021_lsoa = pd.merge(lsoa2msoa, inc2021, left_on='msoacd', right_on='msoacd', how='inner')

# Tidy up
inc2021_lsoa.drop(['msoacd', 'msoanm', 'Local authority code',
                   'Local authority name', 'Region code', 'Region name', 'Upper confidence limit',
                   'Lower confidence limit', 'Confidence interval'], axis=1, inplace=True)

# Sanity check
print("Have " + str(inc2021_lsoa.shape[0]) + " rows data.")
print("Done.")
inc2021_lsoa.sample(3, random_state=r_state)

print('Saving...')
# Save all
inc2011_lsoa.to_csv(os.path.join(housing, 'Private-Income2-2011.csv'), index=False)
inc2021_lsoa.to_csv(os.path.join(housing, 'Private-Income2-2021.csv'), index=False)
print('Saved')

# %% Fare zone
mfz = pd.read_csv(os.path.join(combined, 'MyLondon_fare_zone_OA.csv'))

# Add on LSOA data
lsoa2oa = pd.read_csv(os.path.join(lkp, 'LSOA_OA_2011_JR.csv'))
mfz = pd.merge(lsoa2oa, mfz, left_on='oacd', right_on='OA11CD', how='inner')

# Tidy up
mfz.drop(['oacd', 'OA11CD'], axis=1, inplace=True)

# Calculate mean travel time for the LSOA from all OAs
mfz_lsoa = mfz.groupby('lsoacd').mean()

# Save it
mfz_lsoa.to_csv(os.path.join(social, 'Fare Zone.csv'), index=True)

# Sanity check
print("Have " + str(mfz_lsoa.shape[0]) + " rows data.")
print("Done.")
mfz_lsoa.sample(3, random_state=r_state)
# %% Travel time to Bank
ttb = pd.read_csv(os.path.join(combined, 'MyLondon_traveltime_to_Bank_station_OA.csv'))

# Add on LSOA data
lsoa2oa = pd.read_csv(os.path.join(lkp, 'LSOA_OA_2011_JR.csv'))
ttb = pd.merge(lsoa2oa, ttb, left_on='oacd', right_on='OA11CD', how='inner')

# Tidy up
ttb.drop(['oacd', 'OA11CD'], axis=1, inplace=True)

# Calculate mean travel time for the LSOA from all OAs
ttb_lsoa = ttb.groupby('lsoacd').mean()
ttb_lsoa = ttb_lsoa[['driving_time_mins', 'public_transport_time_mins']]
# Save it
ttb_lsoa.to_csv(os.path.join(social, 'Travel Time To Bank.csv'), index=True)

# Sanity check
print("Have " + str(ttb_lsoa.shape[0]) + " rows of data.")
print("Done.")
ttb_lsoa.sample(3, random_state=r_state)
# %% Work Data
print("Processing occupations files ...")
occ2011 = pd.read_csv(os.path.join(folder2011, 'census2011-occupation.csv'))

# Rename Columns
occ2011.rename(columns={
    '2011 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'All categories: Occupation': 'Total',
    '8. Process plant and machine operatives': '8. Process, plant and machine operatives'
}, inplace=True)

new_names = [re.sub(r'^\d+\. ', '', name) for name in occ2011.columns]
occ2011.columns = new_names

occ2011.drop(['lsoanm'], axis=1, inplace=True)
occ2011 = occ2011[occ2011.lsoacd.isin(ldn2011.lsoacd.values)]

# Sanity check,
print("Have " + str(occ2011.shape[0]) + " rows of data.")
occ2011.sample(3, random_state=r_state)

occ2021 = pd.read_csv(os.path.join(folder2021, 'census2021-occupation-lsoa.csv'))
occ2021.drop(['date'], axis=1, inplace=True)
occ2021.rename(columns=lambda x: re.sub('Occupation (current): ', '', x), inplace=True)

# Rename Columns
occ2021.rename(columns={
    'geography': 'lsoanm',
    'geography code': 'lsoacd',
    'Total: All usual residents aged 16 years and over in employment the week before the census': 'Total',
}, inplace=True)

new_names = [re.sub(r'^\d+\. ', '', name) for name in occ2021.columns]
occ2021.columns = new_names

occ2021 = occ2021[occ2021.lsoacd.isin(ldn2021.lsoacd.values)]
#
occ2021.drop(['lsoanm'], axis=1, inplace=True)

# Sanity check, should be 4994
print("Have " + str(occ2021.shape[0]) + " rows of data.")
occ2021.sample(3, random_state=r_state)

check_dataframe_columns(occ2011, occ2021)

print('Saving...')
# Save all
occ2011.to_csv(os.path.join(work, 'Occupation-2011.csv'), index=False)
occ2021.to_csv(os.path.join(work, 'Occupation-2021.csv'), index=False)
occ2021_conv = convert_lsoa21_to_lsoa11(occ2021, lookup_table, lsoas2011_data)
occ2021_conv.to_csv(os.path.join(work, 'Occupation-Converted-2021.csv'), index=False)

print('Saved')
# %%
ea2011 = pd.read_csv(os.path.join(folder2011, 'census2011-economic-activity-lsoa.csv'))

# Rename Columns
ea2011.rename(columns=lambda x: re.sub('Economically (?:in)?active: ', '', x), inplace=True)
ea2011.rename(columns={
    '2011 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'All categories: Economic activity': 'total',
    'Employee: Part-time': 'PT Employee',
    'Employee: Full-time': 'FT Employee',
    'Self-employed with employees: Part-time': 'PT Self-Employed with Employees',
    'Self-employed with employees: Full-time': 'FT Self-Employed with Employees',
    'Self-employed without employees: Part-time': 'PT Self-Employed without Employees',
    'Self-employed without employees: Full-time': 'FT Self-Employed without Employees',
    'Total': 'Economically inactive'
}, inplace=True)

ea2011.drop(['lsoanm', 'Economically inactive'], axis=1, inplace=True)
# Drop unnecessary columns or they are incompatible with 2011 data
ea2011 = ea2011[ea2011.lsoacd.isin(ldn2011.lsoacd.values)]

# Sample
ea2011.sample(3, random_state=r_state)

## 2021
ea2021 = pd.read_csv(os.path.join(folder2021, 'census2021-economic-activity-lsoa.csv'))
ea2021.rename(columns=lambda x: re.sub('Economically inactive: ', '', x), inplace=True)

ea2021.rename(columns={
    '2021 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'Economically active and a full-time student: Unemployed': 'Full-time student',
    'Student': 'Student (including full-time students)',
    'Total: All usual residents aged 16 years and over': 'total',
}, inplace=True)

ea2021.drop(['Economically active (excluding full-time students):In employment:Self-employed with employees',
             'Economically active (excluding full-time students):In employment:Self-employed without employees',
             'Economically active and a full-time student:In employment:Self-employed with employees',
             'Economically active and a full-time student:In employment:Self-employed without employees',
             'Economically active (excluding full-time students):In employment:Employee',
             'Economically active (excluding full-time students):In employment',
             'Economically active and a full-time student:In employment',
             'Economically active and a full-time student:In employment:Employee'
             ], axis=1, inplace=True)

# Choose boroughs
ea2021 = ea2021[ea2021.lsoacd.isin(ldn2021.lsoacd.values)]
ea2021.drop(['lsoanm'], axis=1, inplace=True)

# Remove descriptions to match column names
ea2021.rename(
    columns=lambda x: re.sub('Economically active \(excluding full-time students\)\: In employment\: ', '', x),
    inplace=True)
ea2021.rename(columns=lambda x: re.sub('Economically active \(excluding full-time students\)\: ', '', x), inplace=True)
ea2021.rename(columns=lambda x: re.sub('Economically active and a full-time student: In employment: ', '', x),
              inplace=True)

# Group the DataFrame by column names and aggregate using sum()
ea2021new = ea2021.groupby(ea2021.columns, axis=1).sum()

# Rename to match 2011
ea2021new.rename(columns={
    'Employee: Part-time': 'PT Employee',
    'Employee: Full-time': 'FT Employee',
    'Self-employed with employees: Part-time': 'PT Self-Employed with Employees',
    'Self-employed with employees: Full-time': 'FT Self-Employed with Employees',
    'Self-employed without employees: Part-time': 'PT Self-Employed without Employees',
    'Self-employed without employees: Full-time': 'FT Self-Employed without Employees',
    'Student': 'Student (including full-time students)',
    'Total': 'Economically inactive'
}, inplace=True)

# Sanity check, should be 4994
print("Have " + str(ea2021.shape[0]) + " rows of data.")
ea2021new.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(ea2011, ea2021new)

print('Saving...')
# Save all
ea2011.to_csv(os.path.join(work, 'Economic Activity-2011.csv'), index=False)
ea2021new.to_csv(os.path.join(work, 'Economic Activity-2021.csv'), index=False)
ea2021conv = convert_lsoa21_to_lsoa11(ea2021new, lookup_table, lsoas2011_data)
ea2021conv.to_csv(os.path.join(work, 'Economic Activity-Converted-2021.csv'), index=False)

print('Saved')
# %% 2011
print("Processing Qualifications data ...")

# Load the data from the KS501EW table
quals_11 = pd.read_csv(os.path.join(folder2011, "census2011-qualification-lsoa.csv"))

# Rename the columns to something easier to work with
quals_11.rename(columns=lambda x: re.sub("(?:Highest level of qualification: )(.+) qualifications", "\\1", x),
                inplace=True)
quals_11.rename(
    columns=lambda x: re.sub("(?:Full-time students: Age 18 to 74: Economically )(?:active: )?(.+)", "Students: \\1",
                             x), inplace=True)
quals_11.rename(columns={'mnemonic': 'lsoacd',
                         '2011 super output area - lower layer': 'LSOANM',
                         'All categories: Highest level of qualification': 'Total'}, inplace=True)

# Select only those rows that are in the London 2011 LSOA list
quals_11 = quals_11.loc[quals_11.lsoacd.isin(ldn2011.lsoacd.values)]

quals_11.drop(['LSOANM'], axis=1, inplace=True)

# Sanity check
print("Wrote " + str(quals_11.shape[0]) + " rows to output file.")
print("Done.")

quals_11.sample(3, random_state=r_state)

## 2021
# Load data
quals_21 = pd.read_csv(os.path.join(folder2021, "census2021-qualification-lsoa.csv"))
quals_21.rename(columns=lambda x: re.sub("Highest level of qualification: ", "", x), inplace=True)
quals_21.rename(columns={'geography code': 'lsoacd',
                         'geography': 'LSOANM',
                         'Total: All usual residents aged 16 years and over': 'Total',
                         'Level 1 and entry level qualifications': 'Level 1 qualifications'
                         }, inplace=True)

# Rename the columns to something easier to work with
quals_21.rename(columns=lambda x: re.sub("(?:Highest level of qualification: )(.+) qualifications", "\\1", x),
                inplace=True)
quals_21.rename(
    columns=lambda x: re.sub("(?:Full-time students: Age 18 to 74: Economically )(?:active: )?(.+)", "Students: \\1",
                             x), inplace=True)

# Select only those rows that are in the London 2021 LSOA list
quals_21 = quals_21.loc[quals_21.lsoacd.isin(ldn2021.lsoacd.values)]

# drop unnecessary columns
quals_21.drop(['date', 'LSOANM'], axis=1, inplace=True)

# Sanity check
print("Wrote " + str(quals_21.shape[0]) + " rows to output file.")
print("Done.")
quals_21.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(quals_11, quals_21)
print('Saving...')
# Save all
quals_11.to_csv(os.path.join(work, 'Qualifications-2011.csv'), index=False)
quals_21.to_csv(os.path.join(work, 'Qualifications-2021.csv'), index=False)
quals_21conv = convert_lsoa21_to_lsoa11(quals_21, lookup_table, lsoas2011_data)
quals_21conv.to_csv(os.path.join(work, 'Qualifications-Converted-2021.csv'), index=False)
print('Saved')

# check_plots(quals_11,quals_21)

# %%
# Industry
ind2011 = pd.read_csv(os.path.join(folder2011, 'census2011-industry-lsoa.csv'))
ind2011.rename(columns={
    '2011 super output area - lower layer': 'lsoa11nm',
    'mnemonic': 'lsoacd',
    'All categories: Industry': 'total'
}, inplace=True)

ind2011.drop(['lsoa11nm'], axis=1, inplace=True)
ind2011 = ind2011[ind2011.lsoacd.isin(ldn2011.lsoacd.values)]
# Aggregate/rename as needed for compatibility
ind2011agg = pd.DataFrame()
ind2011agg['lsoacd'] = ind2011[['lsoacd']]
ind2011agg['A, B, D, E Agriculture, energy and water'] = ind2011[
    [x for x in ind2011.columns if re.search('^[ABDE] ', x)]].sum(axis=1)
ind2011agg['C Manufacturing'] = ind2011[[x for x in ind2011.columns if re.search('^[C] ', x)]].sum(axis=1)
ind2011agg['F Construction'] = ind2011[[x for x in ind2011.columns if re.search('^[F] ', x)]].sum(axis=1)
ind2011agg['G, I Distribution, hotels and restaurants'] = ind2011[
    [x for x in ind2011.columns if re.search('^[GI] ', x)]].sum(axis=1)
ind2011agg['H, J Transport and communication'] = ind2011[[x for x in ind2011.columns if re.search('^[HJ] ', x)]].sum(
    axis=1)
ind2011agg['K, L, M, N Financial, real estate, professional and administrative activities'] = ind2011[
    [x for x in ind2011.columns if re.search('^[KLMN] ', x)]].sum(axis=1)
ind2011agg['O, P, Q Public administration, education and health'] = ind2011[
    [x for x in ind2011.columns if re.search('^[OPQ] ', x)]].sum(axis=1)
ind2011agg['R, S, T, U Other'] = ind2011[[x for x in ind2011.columns if re.search('^[RSTU] ', x)]].sum(axis=1)
ind2011agg['total'] = ind2011[['total']]

# Sanity check, should be 4835
print("Have " + str(ind2011agg.shape[0]) + " rows of data.")
ind2011agg.sample(3, random_state=r_state)

# 2021
ind2021 = pd.read_csv(os.path.join(folder2021, 'census2021-industry-lsoa.csv'))
ind2021.rename(columns={
    '2021 super output area - lower layer': 'lsoa21nm',
    'mnemonic': 'lsoacd',
    'Total': 'total'
}, inplace=True)

ind2021.drop(['lsoa21nm'], axis=1, inplace=True)
ind2021 = ind2021[ind2021.lsoacd.isin(ldn2021.lsoacd.values)]
# Sanity check, should be 4994
print("Have " + str(ind2021.shape[0]) + " rows of data.")
ind2021.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(ind2011agg, ind2021)

print('Saving...')
# Save all
# Save it
ind2011agg.to_csv(os.path.join(work, 'Industry-2011.csv'), index=False)
ind2021.to_csv(os.path.join(work, 'Industry-2021.csv'), index=False)
ind2021conv = convert_lsoa21_to_lsoa11(ind2021, lookup_table, lsoas2011_data)
ind2021conv.to_csv(os.path.join(work, 'Industry-Converted-2021.csv'), index=False)
print('Saved')

# check_plots(ind2011agg,ind2021)
# %%
#### 2011
print("Processing Hours Worked data ...")

# Load the data from the KS501EW table
hours_11 = pd.read_csv(os.path.join(folder2011, "census2011-hours-worked-lsoa.csv"))
# Rename the columns to something easier to work with
hours_11.rename(columns={'mnemonic': 'lsoacd',
                         '2011 super output area - lower layer': 'LSOANM',
                         'All usual residents aged 16 to 74 in employment the week before the census': 'Total'},
                inplace=True)

# Select only those rows that are in the London 2011 LSOA list
hours_11 = hours_11.loc[hours_11.lsoacd.isin(ldn2011.lsoacd.values)]
hours_11.drop(['LSOANM'], axis=1, inplace=True)

# Sanity check
print("Wrote " + str(hours_11.shape[0]) + " rows to output file.")
print("Done.")

hours_11.sample(3, random_state=r_state)

#### 2021
# Load data
hours_21 = pd.read_csv(os.path.join(folder2021, "census2021-hours-worked-lsoa.csv"))
hours_21.rename(columns=lambda x: re.sub("Hours worked: ", "", x), inplace=True)
hours_21.rename(columns={'geography code': 'lsoacd',
                         'geography': 'LSOANM',
                         'Total: All usual residents aged 16 years and over in employment the week before the census': 'Total'
                         }, inplace=True)
# Select only those rows that are in the London 2021 LSOA list
hours_21 = hours_21.loc[hours_21.lsoacd.isin(ldn2021.lsoacd.values)]
# drop unnecessary columns
hours_21.drop(['date', 'LSOANM', 'Part-time', 'Full-time'], axis=1, inplace=True)

# Sanity check
print("Wrote " + str(hours_21.shape[0]) + " rows to output file.")
print("Done.")
hours_21.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(hours_11, hours_21)

print('Saving...')
# Save all
hours_11.to_csv(os.path.join(work, 'Hours Worked-2011.csv'), index=False)
hours_21.to_csv(os.path.join(work, 'Hours Worked-2021.csv'), index=False)
hours_21conv = convert_lsoa21_to_lsoa11(hours_21, lookup_table, lsoas2011_data)
hours_21conv.to_csv(os.path.join(work, 'Hours Worked-Converted-2021.csv'), index=False)
print('Saved')

# check_plots(hours_11,hours_21)

# %%
#### 2011
print("Processing NS-SeC data ...")

# Load the data
ns_sec_11 = pd.read_csv(os.path.join(folder2011, "census2011-NS-SeC -lsoa.csv"))

# Rename the columns to something easier to work with
ns_sec_11.rename(columns={'mnemonic': 'lsoacd',
                          '2011 super output area - lower layer': 'LSOANM',
                          'All categories: NS-SeC': 'Total',
                          '1. Higher managerial, administrative and professional occupations': 'Higher managerial',
                          '2. Lower managerial, administrative and professional occupations': 'Lower managerial',
                          '3. Intermediate occupations': 'Intermediate occupations',
                          '4. Small employers and own account workers': 'Small employers',
                          '5. Lower supervisory and technical occupations': 'Lower supervisory',
                          '6. Semi-routine occupations': 'Semi-routine',
                          '7. Routine occupations': 'Routine',
                          '8. Never worked and long-term unemployed': 'Unemployed',
                          'L15 Full-time students': 'Full-time students'
                          }, inplace=True)

# Select only those rows that are in the London 2011 LSOA list
ns_sec_11 = ns_sec_11.loc[ns_sec_11.lsoacd.isin(ldn2011.lsoacd.values)]
ns_sec_11.drop(['LSOANM'], axis=1, inplace=True)

# Sanity check
print("Wrote " + str(ns_sec_11.shape[0]) + " rows to output file.")
print("Done.")

ns_sec_11.sample(3, random_state=r_state)

#### 2021
# Load data
ns_sec_21 = pd.read_csv(os.path.join(folder2021, "census2021-NS-SeC-lsoa.csv"))
ns_sec_21.rename(columns=lambda x: re.sub('National Statistics Socio-economic Classification \(NS-SEC\): ', '', x),
                 inplace=True)
ns_sec_21.rename(columns={'geography code': 'lsoacd',
                          'geography': 'LSOANM',
                          'Total: All usual residents aged 16 years and over': 'Total',
                          'L1, L2 and L3 Higher managerial, administrative and professional occupations': 'Higher managerial',
                          'L4, L5 and L6 Lower managerial, administrative and professional occupations': 'Lower managerial',
                          'L7 Intermediate occupations': 'Intermediate occupations',
                          'L8 and L9 Small employers and own account workers': 'Small employers',
                          'L10 and L11 Lower supervisory and technical occupations': 'Lower supervisory',
                          'L12 Semi-routine occupations': 'Semi-routine',
                          'L13 Routine occupations': 'Routine',
                          'L14.1 and L14.2 Never worked and long-term unemployed': 'Unemployed',
                          'L15 Full-time students': 'Full-time students'
                          }, inplace=True)
# Select only those rows that are in the London 2021 LSOA list
ns_sec_21 = ns_sec_21.loc[ns_sec_21.lsoacd.isin(ldn2021.lsoacd.values)]
# drop unnecessary columns
ns_sec_21.drop(['date', 'LSOANM'], axis=1, inplace=True)
# Sanity check
print("Wrote " + str(ns_sec_21.shape[0]) + " rows to output file.")
print("Done.")
ns_sec_21.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(ns_sec_11, ns_sec_21)

print('Saving...')
# Save all
ns_sec_11.to_csv(os.path.join(work, 'ns_sec-2011.csv'), index=False)
ns_sec_21.to_csv(os.path.join(work, 'ns_sec-2021.csv'), index=False)
ns_sec_21conv = convert_lsoa21_to_lsoa11(ns_sec_21, lookup_table, lsoas2011_data)
ns_sec_21conv.to_csv(os.path.join(work, 'ns_sec-Converted-2021.csv'), index=False)
print('Saved')

# check_plots(ns_sec_11,ns_sec_21)


# %%
# Using the QS703EW table...
ttw2011 = pd.read_csv(os.path.join(folder2011, 'census2011-travel-method.csv'))
# Convert the column names to something more tractable
ttw2011.rename(columns=lambda x: re.sub("; measures: Value", "", re.sub("Method of Travel to Work: ", "", x)),
               inplace=True)
ttw2011.rename(columns={
    'mnemonic': 'lsoacd',
    '2011 super output area - lower layer': 'lsoanm',
    'Work mainly at or from home': 'Homeworker',
    'Underground, metro, light rail, tram': 'Tube or Tram',
    'Train': 'Train',
    'Bus, minibus or coach': 'Bus',
    'Motorcycle, scooter or moped': 'Motorcycle or Moped',
    'Driving a car or van': 'Private Vehicle',
    'Passenger in a car or van': 'Passenger in Private Vehicle',
    'Bicycle': 'Bicycle',
    'On foot': 'Foot',
    'Other method of travel to work': 'Other travel method'
}, inplace=True)

ttw2011.sample(3, random_state=r_state)

# Drop the non-London LSOAs
ttw2011.drop(['lsoanm', 'Not in employment'], axis=1, inplace=True)
ttw2011 = ttw2011[ttw2011.lsoacd.isin(ldn2011.lsoacd.values)]

# Sanity check
print("2011 TTW data frame contains " + str(ttw2011.shape[0]) + " rows.")
ttw2011.sample(3, random_state=r_state)

# Using the QS703EW table...
ttw2021 = pd.read_csv(os.path.join(folder2021, 'census2021-travel-method-lsoa.csv'))

# Convert the column names to something more tractable
ttw2021.rename(columns=lambda x: re.sub('Method of travel to workplace: ', "", x), inplace=True)
ttw2021.rename(columns={
    'geography code': 'lsoacd',
    'geography': 'lsoanm',
    'Total: All usual residents aged 16 years and over in employment the week before the census': 'Total',
    'Work mainly at or from home': 'Homeworker',
    'Underground, metro, light rail, tram': 'Tube or Tram',
    'Train': 'Train',
    'Bus, minibus or coach': 'Bus',
    'Motorcycle, scooter or moped': 'Motorcycle or Moped',
    'Driving a car or van': 'Private Vehicle',
    'Passenger in a car or van': 'Passenger in Private Vehicle',
    'Bicycle': 'Bicycle',
    'On foot': 'Foot',
    'Other method of travel to work': 'Other travel method'
}, inplace=True)

ttw2021.sample(3, random_state=r_state)

# Drop the non-London LSOAs
ttw2021.drop(['date', 'lsoanm', 'Total'], axis=1, inplace=True)
ttw2021 = ttw2021[ttw2021.lsoacd.isin(ldn2021.lsoacd.values)]

# Sanity check
print("2021 TTW data frame contains " + str(ttw2021.shape[0]) + " rows.")
ttw2021.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(ttw2011, ttw2021)

print('Saving...')
# Save all
ttw2011.to_csv(os.path.join(work, 'Travel Methods-2011.csv'), index=False)
ttw2021.to_csv(os.path.join(work, 'Travel Methods-2021.csv'), index=False)
ttw2021conv = convert_lsoa21_to_lsoa11(ttw2021, lookup_table, lsoas2011_data)
ttw2021conv.to_csv(os.path.join(work, 'Travel Methods-Converted-2021.csv'), index=False)
print('Saved')

# check_plots(ttw2011,ttw2021)

# %% Number of cars
cv2011 = pd.read_csv(os.path.join(folder2011, 'census2011-cars-lsoa.csv'))

# More analysis-friendly column names
cv2011.rename(columns={
    '2011 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'All categories: Car or van availability': 'total',
    'No cars or vans in household': 'No vehicle',
    '1 car or van in household': '1 vehicle',
    '2 cars or vans in household': '2 vehicles',
    '3 cars or vans in household': '3 vehicles',
    '4 or more cars or vans in household': '4 or more vehicles',
}, inplace=True)

# Don't need these

cv2011['3 vehicles or more vehicles'] = cv2011['3 vehicles'] + cv2011['4 or more vehicles']
cv2011.drop(['lsoanm', '3 vehicles', '4 or more vehicles'], axis=1, inplace=True)
# Drop the rows we don't need
cv2011 = cv2011[cv2011.lsoacd.isin(ldn2011.lsoacd.values)]
# Sanity check, should be 4835
print("Have " + str(cv2011.shape[0]) + " rows of data.")
cv2011.sample(3, random_state=r_state)

cv2021 = pd.read_csv(os.path.join(folder2021, 'census2021-cars-lsoa.csv'))
cv2021.rename(columns=lambda x: re.sub('Number of cars or vans: ', "", x), inplace=True)

# More analysis-friendly column names
cv2021.rename(columns={
    'geography': 'lsoanm',
    'geography code': 'lsoacd',
    'Total: All households': 'total',
    'No cars or vans in household': 'No vehicle',
    '1 car or van in household': '1 vehicle',
    '2 cars or vans in household': '2 vehicles',
    '3 or more cars or vans in household': '3 vehicles or more vehicles'
}, inplace=True)

# Don't need these
cv2021.drop(['lsoanm'], axis=1, inplace=True)

# Drop the rows we don't need
cv2021 = cv2021[cv2021.lsoacd.isin(ldn2021.lsoacd.values)]
# Sanity check, should be 4835
print("Have " + str(cv2021.shape[0]) + " rows of data.")
cv2021.sample(3, random_state=r_state)

check_dataframe_columns(cv2011, cv2021)

# Save it
cv2011.to_csv(os.path.join(housing, 'Cars and Vans-2011.csv'), index=False)
cv2021.to_csv(os.path.join(housing, 'Cars and Vans-2011.csv'), index=False)
cv2021conv = convert_lsoa21_to_lsoa11(cv2021, lookup_table, lsoas2011_data)
cv2021conv.to_csv(os.path.join(housing, 'Cars and Vans-Converted-2021.csv'), index=False)
print('Saved')
# %%
#### 2011
print("Processing Population Density data ...")
den2011 = pd.read_csv(os.path.join(folder2011, 'census2011-population-density-lsoa.csv'))

# Change column names
den2011.rename(columns={
    '2011 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'Density (number of persons per hectare)': 'Density'
}, inplace=True)

# Drop unnecessary columns
den2011.drop(['lsoanm'], axis=1, inplace=True)
den2011['Density'] = den2011['Density'] * 100
den2011 = den2011[den2011.lsoacd.isin(ldn2011.lsoacd.values)]

# Sanity check,
print("Have " + str(den2011.shape[0]) + " rows of data.")
den2011.sample(3, random_state=r_state)

## 2021
den2021 = pd.read_csv(os.path.join(folder2021, 'census2021-population-density-lsoa.csv'))

# Change column names
den2021.rename(columns={
    'geography': 'lsoanm',
    'geography code': 'lsoacd',
    'Population Density: Persons per square kilometre; measures: Value': 'Density'
}, inplace=True)

# Drop unnecessary columns
den2021.drop(['lsoanm', 'date'], axis=1, inplace=True)
den2021 = den2021[den2021.lsoacd.isin(ldn2021.lsoacd.values)]

# Sanity check, should be 4994
print("Have " + str(den2021.shape[0]) + " rows of data.")
den2021.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(den2011, den2021)

print('Saving...')
# Save all
den2011.to_csv(os.path.join(demographics, 'Density-2011.csv'), index=False)
den2021.to_csv(os.path.join(demographics, 'Density-2021.csv'), index=False)
den2021_conv = convert_lsoa21_to_lsoa11(den2021, lookup_table, lsoas2011_data)
den2021_conv.to_csv(os.path.join(demographics, 'Density-Converted-2021.csv'), index=False)
print('Saved')

# check_plots(den2011,den2021)


# %%
### 2011
print("Processing Household Composition data ...")
hhc2011 = pd.read_csv(os.path.join(folder2011, 'census2011-household-composition-lsoa.csv'))

# Rename Columns
hhc2011.rename(columns={
    '2011 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'All categories: Household composition': 'total'
}, inplace=True)

# Rename specific columns using regular expressions to shorten the names
hhc2011.rename(columns=lambda x: re.sub('One person household(?:\: )?', '1P: ', x), inplace=True)
hhc2011.rename(columns=lambda x: re.sub('One family only(?:\: )?', '1F: ', x), inplace=True)
hhc2011.rename(columns=lambda x: re.sub(': Married or same-sex civil partnership couple(?:\: )?', 'M: ', x),
               inplace=True)
hhc2011.rename(columns=lambda x: re.sub(': Cohabiting couple(?:\: )?', 'C: ', x), inplace=True)
hhc2011.rename(columns=lambda x: re.sub(': Lone parent(?:\: )?', 'L: ', x), inplace=True)
hhc2011.rename(columns=lambda x: re.sub('Other household types(?:\: )?', 'O: ', x), inplace=True)
hhc2011.rename(columns=lambda x: re.sub('A(?:ll a)?ged 65 and over', '65+', x), inplace=True)
hhc2011.rename(columns=lambda x: re.sub('All full-time students', 'Students', x), inplace=True)

# Combine households with students and those aged 65+ into a single category
hhc2011['O: Students and 65+'] = hhc2011['O: Students'] + hhc2011['O: 65+']
# Drop unnecessary columns
hhc2011.drop(['lsoanm', 'One family household', '1P: ', '1FM: ', '1FC: ', '1FL: ', 'O: ', 'O: Other',
              'O: Students',
              'O: 65+'], axis=1, inplace=True)

# Keep only the LSOAs that are present in the London 2011 dataset
hhc2011 = hhc2011[hhc2011.lsoacd.isin(ldn2011.lsoacd.values)]
print("Have " + str(hhc2011.shape[0]) + " rows of data.")
hhc2011.sample(3, random_state=r_state)

## 2021
hhc2021 = pd.read_csv(os.path.join(folder2021, 'census2021-household-composition-lsoa.csv'))
hhc2021.rename(columns=lambda x: re.sub('Household composition: ', '', x), inplace=True)
hhc2021.rename(columns=lambda x: re.sub('; measures: Value', '', x), inplace=True)

# Rename columns
hhc2021.rename(columns={
    'geography': 'lsoanm',
    'geography code': 'lsoacd'
}, inplace=True)

# Rename specific columns using regular expressions to shorten the names
hhc2021.rename(columns=lambda x: re.sub('One person household(?:\: )?', '1P: ', x), inplace=True)
hhc2021.rename(columns=lambda x: re.sub('Single family household(?:\: )?', '1F: ', x), inplace=True)
hhc2021.rename(columns=lambda x: re.sub(': Married or civil partnership couple(?:\: )?', 'M: ', x), inplace=True)
hhc2021.rename(columns=lambda x: re.sub(': Cohabiting couple family(?:\: )?', 'C: ', x), inplace=True)
hhc2021.rename(columns=lambda x: re.sub(': Lone parent family(?:\: )?', 'L: ', x), inplace=True)
hhc2021.rename(columns=lambda x: re.sub('Other household types(?:\: )?', 'O: ', x), inplace=True)
hhc2021.rename(columns=lambda x: re.sub('A(?:ll a)?ged 66 years and over', '65+', x), inplace=True)
hhc2021.rename(columns=lambda x: re.sub('Other, including all full-time students and all aged 66 years and over',
                                        'Students and 65+', x), inplace=True)

hhc2021.rename(columns=({'1FC: With dependent children': '1FC: Dependent children',
                         '1FL: With dependent children': '1FL: Dependent children',
                         'Total': 'total'
                         }), inplace=True)

# Drop unnecessary columns
hhc2021.drop(['date', 'lsoanm', '1F: ', '1P: ', '1FM: ', '1FC: ', '1FL: ', 'O: ',
              '1F: Other single family household',
              '1F: Other single family household: Other family composition'], axis=1, inplace=True)
hhc2021 = hhc2021[hhc2021.lsoacd.isin(ldn2021.lsoacd.values)]

# Sanity check, should be 4994
print("Have " + str(hhc2021.shape[0]) + " rows of data.")
hhc2021.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(hhc2011, hhc2021)

print('Saving...')
# Save all
hhc2011.to_csv(os.path.join(demographics, 'Household Composition-2011.csv'), index=False)
hhc2021.to_csv(os.path.join(demographics, 'Household Composition-2021.csv'), index=False)
hhc2021_conv = convert_lsoa21_to_lsoa11(hhc2021, lookup_table, lsoas2011_data)
hhc2021_conv.to_csv(os.path.join(demographics, 'Household Composition-Converted-2021.csv'), index=False)
print('Saved')

# check_plots(hhc2011,hhc2021)


# %%
print("Processing Age Composition data ...")
age2011 = pd.read_csv(os.path.join(folder2011, 'census2011-age-structure-lsoa.csv'))  # , na_values="..")

# Rename Columns
age2011.rename(columns={
    '2011 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'All usual residents': 'total'
}, inplace=True)

# Drop unnecessary columns
age2011.drop(['lsoanm'], axis=1, inplace=True)
age2011 = age2011[age2011.lsoacd.isin(ldn2011.lsoacd.values)]

# Aggregate into a new data frame
age2011agg = pd.DataFrame()

# Drop unnecessary columns
age2011agg['0 to 9'] = age2011[['Age 0 to 4', 'Age 5 to 7', 'Age 8 to 9']].sum(axis=1)
age2011agg['10 to 19'] = age2011[['Age 10 to 14', 'Age 15', 'Age 16 to 17', 'Age 18 to 19']].sum(axis=1)
age2011agg['20 to 29'] = age2011[['Age 20 to 24', 'Age 25 to 29']].sum(axis=1)
age2011agg['30 to 59'] = age2011[['Age 30 to 44', 'Age 45 to 59']].sum(axis=1)
age2011agg['60 to 74'] = age2011[['Age 60 to 64', 'Age 65 to 74']].sum(axis=1)
age2011agg['75 to 90+'] = age2011[['Age 75 to 84', 'Age 85 to 89', 'Age 90 and over']].sum(axis=1)
age2011agg['total'] = age2011[['total']]
age2011agg['lsoacd'] = age2011[['lsoacd']]

print("Have " + str(age2011agg.shape[0]) + " rows of data.")
age2011agg.sample(3, random_state=r_state)

age2021 = pd.read_csv(os.path.join(folder2021, 'census2021-age-structure-lsoa.csv'))
age2021.rename(columns=lambda x: re.sub('Age: ', '', x), inplace=True)

age2021.drop(['date'], axis=1, inplace=True)

# Rename Columns
age2021.rename(columns={
    'geography': 'lsoanm',
    'geography code': 'lsoacd',
    'Total': 'total'
}, inplace=True)

# Drop unnecessary columns
age2021.drop(['lsoanm'], axis=1, inplace=True)
age2021 = age2021[age2021.lsoacd.isin(ldn2021.lsoacd.values)]

# Aggregate into a new data frame
age2021agg = pd.DataFrame()

# Drop unnecessary columns
age2021agg['0 to 9'] = age2021[['Aged 4 years and under', 'Aged 5 to 9 years']].sum(axis=1)
age2021agg['10 to 19'] = age2021[['Aged 10 to 14 years', 'Aged 15 to 19 years']].sum(axis=1)
age2021agg['20 to 29'] = age2021[['Aged 20 to 24 years', 'Aged 25 to 29 years']].sum(axis=1)
age2021agg['30 to 59'] = age2021[['Aged 30 to 34 years', 'Aged 35 to 39 years', 'Aged 40 to 44 years',
                                  'Aged 45 to 49 years', 'Aged 50 to 54 years', 'Aged 55 to 59 years']].sum(axis=1)
age2021agg['60 to 74'] = age2021[['Aged 60 to 64 years', 'Aged 65 to 69 years', 'Aged 70 to 74 years']].sum(axis=1)
age2021agg['75 to 90+'] = age2021[['Aged 75 to 79 years', 'Aged 80 to 84 years', 'Aged 85 years and over']].sum(axis=1)
age2021agg['total'] = age2021[['total']]
age2021agg['lsoacd'] = age2021[['lsoacd']]

# Sanity check,
print("Have " + str(age2021agg.shape[0]) + " rows of data.")
age2021agg.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(age2011agg, age2021agg)

print('Saving...')
# Save
age2011agg.to_csv(os.path.join(demographics, 'Age Structure-2011.csv'), index=False)
age2021agg.to_csv(os.path.join(demographics, 'Age Structure-2021.csv'), index=False)
age2021agg_conv = convert_lsoa21_to_lsoa11(age2021agg, lookup_table, lsoas2011_data)
age2021agg_conv.to_csv(os.path.join(demographics, 'Age Structure-Converted-2021.csv'), index=False)
print('Saved')

# check_plots(age2011agg,age2021agg)
# %%
print("Processing Ethnic Group data ...")
eth2011 = pd.read_csv(os.path.join(folder2011, 'census2011-ethnic-group-lsoa.csv'))

# Rename Columns
eth2011.rename(columns={
    '2011 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'All usual residents': 'total',
    'Mixed/multiple ethnic groups': 'Mixed',
    'Black/African/Caribbean/Black British': 'Black',
    'Other ethnic group: Any other ethnic group': 'Other Ethnicity',
    'White: Gypsy or Irish Traveller': 'White: Gypsy/Irish Traveller'
}, inplace=True)

eth2011.rename(columns=lambda x: re.sub('Mixed/multiple ethnic groups: ', 'Mixed: ', x), inplace=True)
eth2011.drop(['lsoanm'], axis=1, inplace=True)
# Drop unnecessary columns
eth2011 = eth2011[eth2011.lsoacd.isin(ldn2011.lsoacd.values)]

# Aggregate these for compatibility with rest of data
eth2011agg = pd.DataFrame()
eth2011agg = eth2011.loc[:, ['lsoacd', 'White', 'Mixed', 'Black', 'Other Ethnicity']]
eth2011agg['Asian'] = eth2011[[x for x in eth2011.columns if re.search('^Asian/Asian British: [^C]', x)]].sum(axis=1)
eth2011agg['Chinese'] = eth2011[[x for x in eth2011.columns if re.search('Chinese$', x)]].sum(axis=1)
eth2011agg['total'] = eth2011[['total']]

print("Have " + str(eth2011.shape[0]) + " rows of data.")
eth2011agg.sample(3, random_state=r_state)

## 2021
eth2021 = pd.read_csv(os.path.join(folder2021, 'census2021-ethnic-group-lsoa.csv'))
eth2021.rename(columns=lambda x: re.sub('Ethnic group: ', '', x), inplace=True)

# Drop column
eth2021.drop(['date', 'White: Roma'], axis=1, inplace=True)

# Rename Column
eth2021.rename(columns={
    'geography': 'lsoanm',
    'geography code': 'lsoacd',
    'Total: All usual residents': 'total',
    'Mixed or Multiple ethnic groups': 'Mixed',
    'Black, Black British, Black Welsh, Caribbean or African': 'Black',
    'Other ethnic group: Any other ethnic group': 'Other Ethnicity',
    'Mixed or Multiple ethnic groups: Other Mixed or Multiple ethnic groups': 'Mixed: Other Mixed'
}, inplace=True)

# Drop unnecessary columns
eth2021 = eth2021[eth2021.lsoacd.isin(ldn2021.lsoacd.values)]
eth2021.drop(['lsoanm'], axis=1, inplace=True)

# Rename columns to match
eth2021.rename(columns=lambda x: re.sub(' or ', '/', x), inplace=True)
eth2021.rename(columns=lambda x: re.sub(', ', '/', x), inplace=True)
eth2021.rename(columns=lambda x: re.sub('/Asian Welsh', '', x), inplace=True)
eth2021.rename(columns=lambda x: re.sub('Black/Black British/Black Welsh/Caribbean/African',
                                        'Black/African/Caribbean/Black British', x), inplace=True)
eth2021.rename(columns=lambda x: re.sub('Mixed/Multiple ethnic groups: ', 'Mixed: ', x), inplace=True)

# Aggregate these for compatibility with rest of data
eth2021agg = pd.DataFrame()
eth2021agg = eth2021.loc[:, ['lsoacd', 'White', 'Mixed', 'Black', 'Other Ethnicity']]
eth2021agg['Asian'] = eth2021[[x for x in eth2021.columns if re.search('^Asian/Asian British: [^C]', x)]].sum(axis=1)
eth2021agg['Chinese'] = eth2021[[x for x in eth2021.columns if re.search('Chinese$', x)]].sum(axis=1)
eth2021agg['total'] = eth2021[['total']]

# Sanity check,
print("Have " + str(eth2021.shape[0]) + " rows of data.")
eth2021agg.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(eth2011agg, eth2021agg)

# Save it
print('Saving...')

# Save
eth2011agg.to_csv(os.path.join(demographics, 'Ethnicity-2011.csv'), index=False)
eth2021agg.to_csv(os.path.join(demographics, 'Ethnicity-2021.csv'), index=False)
eth2021_conv = convert_lsoa21_to_lsoa11(eth2021agg, lookup_table, lsoas2011_data)
eth2021_conv.to_csv(os.path.join(demographics, 'Ethnicity-Converted-2021.csv'), index=False)
print('Saved')

# check_plots(eth2011,eth2021)
# %%
mar2011 = pd.read_csv(os.path.join(folder2011, 'census2011-marital-status-lsoa.csv'))

# More analysis-friendly column names
mar2011.rename(columns={
    '2011 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'All usual residents aged 16+': 'total',
    'Single (never married or never registered a same-sex civil partnership)': 'Single (never married)',
    'Separated (but still legally married or still legally in a same-sex civil partnership)': 'Separated',
    'Divorced or formerly in a same-sex civil partnership which is now legally dissolved': 'Divorced',
    'Widowed or surviving partner from a same-sex civil partnership': 'Widowed',
    'Married': 'Married Temp'
}, inplace=True)

# Aggregate these for compatibility with rest of data
mar2011['Married'] = mar2011[['Married Temp', 'In a registered same-sex civil partnership']].sum(axis=1)

# Don't need these
mar2011.drop(['lsoanm', 'Married Temp', 'In a registered same-sex civil partnership'], axis=1, inplace=True)
mar2011 = mar2011[mar2011.lsoacd.isin(ldn2011.lsoacd.values)]

# Sanity check, should be 4835
print("Have " + str(mar2011.shape[0]) + " rows of data.")
mar2011.sample(3, random_state=r_state)

mar2021 = pd.read_csv(os.path.join(folder2021, 'census2021-marital-status-lsoa.csv'))

# More analysis-friendly column names
mar2021.rename(columns={
    '2021 super output area - lower layer': 'lsoanm',
    'mnemonic': 'lsoacd',
    'All usual residents aged 16+': 'total',
    'Never married and never registered a civil partnership': 'Single (never married)',
    'Separated, but still legally married or still legally in a civil partnership': 'Separated',
    'Divorced or civil partnership dissolved': 'Divorced',
    'Widowed or surviving civil partnership partner': 'Widowed',
    'Married or in a registered civil partnership': 'Married'
}, inplace=True)

# Don't need these
mar2021 = mar2021[['lsoacd', 'Single (never married)', 'Separated', 'Divorced', 'Widowed', 'Married']]

mar2021['total'] = mar2021.sum(axis=1)
mar2021 = mar2021[mar2021.lsoacd.isin(ldn2021.lsoacd.values)]

# Sanity check, should be 4835
print("Have " + str(mar2021.shape[0]) + " rows of data.")
mar2021.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(mar2011, mar2021)

# Save it
print('Saving...')
# Save it
mar2011.to_csv(os.path.join(demographics, 'Marital Status-2011.csv'), index=False)
mar2021.to_csv(os.path.join(demographics, 'Marital Status-2021.csv'), index=True)
mar2021_conv = convert_lsoa21_to_lsoa11(mar2021, lookup_table, lsoas2011_data)
mar2021_conv.to_csv(os.path.join(demographics, 'Marital Status-Converted-2021.csv'), index=False)
print('Saved')

# check_plots(mar2011,mar2021)


# %%
print("Processing Crime data ...")
crime_data = pd.read_csv(os.path.join(combined, 'MPS LSOA Level Crime (Historical).csv'))

# Rename Columns
crime_data.rename(columns={
    'LSOA Name': 'lsoanm',
    'LSOA Code': 'lsoacd'
}, inplace=True)

# Drop unnecessary columns
crime_data = crime_data[crime_data.lsoacd.isin(ldn2021.lsoacd.values)]

# Aggregate into a new data frame
crime_dataagg = pd.DataFrame()
crime_dataagg = crime_data[['lsoacd', 'Major Category']]
crime_dataagg['2011'] = crime_data[[x for x in crime_data.columns if re.search('^2011', x)]].sum(axis=1)
crime_dataagg['2021'] = crime_data[[x for x in crime_data.columns if re.search('^2021', x)]].sum(axis=1)

# Create separate dataframes
crime2011 = crime_dataagg[["lsoacd", "Major Category", "2011"]]
crime2021 = crime_dataagg[["lsoacd", "Major Category", "2021"]]

# Pivot and aggregate
crime2011 = crime2011.pivot_table(index="lsoacd", values="2011", columns="Major Category", aggfunc=sum)
crime2021 = crime2021.pivot_table(index="lsoacd", values="2021", columns="Major Category", aggfunc=sum)
crime2011.reset_index(inplace=True)
crime2011.rename(columns={0: 'lsoacd'}, inplace=True)
crime2021.reset_index(inplace=True)
crime2021.rename(columns={0: 'lsoacd'}, inplace=True)

print("Have " + str(crime2011.shape[0]) + " rows of data.")
crime2011.sample(3, random_state=r_state)

print("Have " + str(crime2021.shape[0]) + " rows of data.")
crime2021.sample(3, random_state=r_state)

# check column names
check_dataframe_columns(crime2011, crime2021)

# %%
crime_2011_conv = convert_lsoa21_to_lsoa11(crime2011, lookup_table, lsoas2011_data)
crime_2021_conv = convert_lsoa21_to_lsoa11(crime2021, lookup_table, lsoas2011_data)

# Define the weights and adjacency for 2011
qw2011 = weights.Queen.from_shapefile(os.path.join('data', 'shp', 'LSOA-2011-Weights.shp'))  # Weights/Adjacency
fh2011 = ps.lib.io.open(os.path.join('data', 'shp', 'LSOA-2011-Weights.dbf'))
cds2011 = fh2011.by_col['LSOA11CD']  # LSOA 2021 Census code

crime_2011_conv = fill_missing_values_2(crime_2011_conv, qw2011, fh2011, cds2011)
crime_2021_conv = fill_missing_values_2(crime_2021_conv, qw2011, fh2011, cds2011)
# %%
check_missing_values(crime_2011_conv)
check_missing_values(crime_2021_conv)
# %%
# Save it
print('Saving...')

# Save
crime_2011_conv.to_csv(os.path.join(social, 'Crime-Converted-2011.csv'), index=False)
crime_2021_conv.to_csv(os.path.join(social, 'Crime-Converted-2021.csv'), index=False)

print('Saved')

# check_plots(crime_2011_conv,crime_2021_conv)
# %%
print("Processing Greenspace ...")
greenspace = pd.read_csv(os.path.join(combined, 'lsoapublicparks.csv'))
# Rename Columns
greenspace.rename(columns={
    'LSOA name': 'lsoanm',
    'LSOA code': 'lsoacd',
    'Average number of  Parks, Public Gardens, or Playing Fields within 1,000 m radius': 'Avg Number of Parks',
    'Average distance to nearest Park, Public Garden, or Playing Field (m)': 'Avg distance'
}, inplace=True)

greenspace = greenspace[['lsoacd', 'Avg Number of Parks', 'Avg distance']]

# Drop unnecessary columns
greenspace = greenspace[greenspace.lsoacd.isin(ldn2011.lsoacd.values)]

greenspace.sample(3, random_state=r_state)
print("Have " + str(greenspace.shape[0]) + " rows of data.")
# Save it
print('Saving...')
# Save
greenspace.to_csv(os.path.join(social, 'Greenspace.csv'), index=False)
print('Saved')
