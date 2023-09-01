# %%
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from sklearn import decomposition
from sklearn import preprocessing

r_state = 42
random.seed(r_state)
np.random.seed(r_state)

pd.set_option('display.max_columns', None)

# %%
# Create folders path
lkp = os.path.join('data', 'lkp')
shp = os.path.join('data', 'shp')
src = os.path.join('data', 'src')
zip = os.path.join('data', 'input', 'zip')
analytical = os.path.join('data', 'analytical')

# Create Directories
factors = os.path.join('data', 'factors')
housing = os.path.join(factors, 'housing')
demographics = os.path.join(factors, 'demographics')
work = os.path.join(factors, 'work')

# %%
if os.path.exists(analytical):
    shutil.rmtree(analytical)

if not os.path.exists(analytical):
    os.makedirs(analytical)

# %%
ldn2011 = pd.read_pickle(os.path.join(lkp, 'LSOAs 2011.pkl'))
ldn2021 = pd.read_pickle(os.path.join(lkp, 'LSOAs 2021.pkl'))

print("Have built London LSOA filter data for use where needed...")
print("\t2011: " + str(ldn2011.shape[0]) + " rows.")
print("\t2021: " + str(ldn2021.shape[0]) + " rows.")


# %%
def plot_checks(df, selected_cols=None):
    sns.set(rc={"figure.figsize": (12, 3)})
    if not selected_cols:
        selected_cols = df.columns
    for d in selected_cols:
        print("Working on " + d)
        sns.set_theme(style="white",
                      palette=None,
                      rc={"axes.spines.right": False, "axes.spines.top": False})
        sns.histplot(df[d], color='green')
        plt.title(d)
        plt.show()
    print("Done.")
    return


# %%
df11 = pd.DataFrame()
df21 = pd.DataFrame()

# for d in ['Private-Income-LSOAs.csv','Private-Rents-LSOAs.csv','Property-Price.csv','Occupation.csv','Qualifications.csv','ns_sec.csv']:
#     housing_folder=['Private-Income-LSOAs','Private-Rent','Property-Price']
for d in ['Private-Income2.csv', 'Private-Rents-LSOAs.csv', 'Property-Price.csv', 'Occupation.csv',
          'Qualifications.csv', 'ns_sec.csv']:
    housing_folder = ['Private-Income2', 'Private-Rent', 'Property-Price']
    if any(word in d for word in housing_folder):
        loc = housing
    else:
        loc = work

    tmp11_df = pd.read_csv(os.path.join(loc, d.replace('.csv', '-2011.csv')))
    if d in ['Property-Price.csv', 'Private-Income2.csv']:
        tmp21_df = pd.read_csv(os.path.join(loc, d.replace('.csv', '-2021.csv')))
    else:
        tmp21_df = pd.read_csv(os.path.join(loc, d.replace('.csv', '-Converted-2021.csv')))

    if df11.shape[0] == 0:
        df11 = tmp11_df
        df21 = tmp21_df
    else:
        df11 = pd.merge(df11, tmp11_df, how='outer', left_on='lsoacd', right_on='lsoacd')
        df21 = pd.merge(df21, tmp21_df, how='outer', left_on='lsoacd', right_on='lsoacd')

print("Shape of 2021 data frame: " + str(df21.shape))
print("Shape of 2011 data frame: " + str(df11.shape))

rename = {
    'Income': 'Median_inc',
    'Median Rent': 'Median_rent',
    'Median Property Price': 'Median_hp',
    'Total_x': 'Total_occupations',
    'Total_y': 'Total_qualifications',
    'Total': 'Total_nssec'

}
df11.rename(columns=rename, inplace=True)
df21.rename(columns=rename, inplace=True)
print("Columns renamed to remove ambiguity.")

#  Set the index of dataframe to LSOA
df11.set_index('lsoacd', inplace=True)
df21.set_index('lsoacd', inplace=True)
print("Datasets indexed to LSOA")

df11.sample(3, random_state=r_state)

# plot_checks(df11)

# %% Log transform house prices
hp21 = np.log(df21['Median_hp'])
hp11 = np.log(df11['Median_hp'])
print("Property prices transformed using natural log.")

hpr21 = np.log(df21['Median_rent'])
hpr11 = np.log(df11['Median_rent'])
print("Property prices transformed using natural log.")

hinc21 = np.log(df21['Median_inc'])
hinc11 = np.log(df11['Median_inc'])
print("Income transformed using natural log.")

df_transformed_log = pd.DataFrame({
    'hp_11': hp11,
    'hp_21': hp21,
    'hpr_11': hpr11,
    'hpr_21': hpr21,
    'hinc_11': hinc11,
    'hinc_21': hinc21
}, index=df11.index)

print("Final shape: " + str(df_transformed_log.shape))
df_transformed_log.describe()

# %%
#  Process occupational data
def process_occ_data(df):
    #  Columns of interest
    occ = ['Managers, directors and senior officials',
           'Professional occupations',
           'Associate professional and technical occupations',
           'Administrative and secretarial occupations',
           'Skilled trades occupations',
           'Caring, leisure and other service occupations',
           'Sales and customer service occupations',
           'Process, plant and machine operatives',
           'Elementary occupations']

    # Calculate the share of that occupation type in relation to the total number of occupations in each row of the DataFrame.
    occ_data = pd.DataFrame()
    # It magnifies the differences between small and large share values.
    # Squaring the share values gives greater weight to occupation types that have a higher share relative to others. This is useful for capturing the relative dominance or concentration of specific occupation types within the dataset.
    for c in occ:
        occ_data[c + '_share'] = (df.loc[:, c] / df.loc[:, 'Total_occupations']) ** 2

    # Drop the share columns
    occ_data.drop([s for s in occ_data.columns if '_share' in s], axis=1, inplace=True)

    # Add the 'knowledge worker' share -- this is columns 0-2 of the data frame
    # It calculates the "knowledge worker" percentage by summing the values of the first
    # three columns (representing managers, directors, senior officials, professional occupations, and associate professional and technical occupations)
    # and dividing it by the total number of occupations.
    # occ_data['kw_pct'] = (df.loc[:, occ[0:3]].sum(axis=1) / df.loc[:, 'Total_occupations']) * 100
    occ_data['kw_pct'] = (df.loc[:, occ[0:3]].sum(axis=1) / df.loc[:, 'Total_occupations'])

    return occ_data


occ11 = process_occ_data(df11)
sns.histplot(occ11, x='kw_pct')
plt.show()

occ11.sample(3, random_state=r_state)


# %%
#  Process qualifications data
def process_quals_data(df):
    # Columns of interest
    quals = ['No qualifications', 'Level 1 qualifications',
             'Level 2 qualifications', 'Apprenticeship', 'Level 3 qualifications',
             'Level 4 qualifications and above', 'Other qualifications']
    squals = ['Students In employment', 'Students Unemployed', 'Students inactive']  # Not currently used

    #  Integrate results into Qualifications datasets  --
    quals_data = pd.DataFrame()

    for c in quals:
        quals_data[c + '_share'] = (df.loc[:, c] / df.loc[:, 'Total_qualifications']) ** 2

    # Drop the share columns
    quals_data.drop([s for s in quals_data.columns if '_share' in s], axis=1, inplace=True)

    # The 'highly educated' share -- this is columns 0-2 of the data frame
    # quals_data['he_pct'] = (df.loc[:, quals[5]] / df.loc[:, 'Total_qualifications']) * 100
    quals_data['he_pct'] = (df.loc[:, quals[5]] / df.loc[:, 'Total_qualifications'])

    return quals_data


qual11 = process_quals_data(df11)
sns.histplot(qual11, x='he_pct')
plt.show()
qual11.sample(3, random_state=r_state)


# %%
def process_nssec_data(df):
    # Columns of interest
    nssec = ['Higher managerial', 'Lower managerial',
             'Intermediate occupations', 'Small employers', 'Lower supervisory',
             'Semi-routine', 'Routine', 'Unemployed', 'Full-time students']

    #  Integrate results into Qualifications datasets  --
    nssec_data = pd.DataFrame()

    for c in nssec:
        nssec_data[c + '_share'] = (df.loc[:, c] / df.loc[:, 'Total_nssec']) ** 2

    # Drop the share columns
    nssec_data.drop([s for s in nssec_data.columns if '_share' in s], axis=1, inplace=True)

    # The 'highly educated' share -- this is columns 0-2 of the data frame
    nssec_data['hs_pct'] = (
                df[['Higher managerial', 'Lower managerial', 'Small employers']].sum(axis=1) / df.loc[:, 'Total_nssec'])
    # nssec_data['hs_pct'] = (df[['Higher managerial','Lower managerial','Small employers','Intermediate occupations']].sum(axis=1) / df.loc[:, 'Total_nssec'])

    # nssec_data['hs_pct'] = (df[['Higher managerial','Small employers']].sum(axis=1) / df.loc[:, 'Total_nssec']) * 100
    # nssec_data['hs_pct'] = (df[['Higher managerial', 'Lower managerial']].sum(axis=1) / df.loc[:, 'Total_nssec'])* 100
    return nssec_data


nssec11 = process_nssec_data(df11)
sns.histplot(nssec11, x='hs_pct')
plt.show()
nssec11.sample(3, random_state=r_state)
# %%
quals_11 = process_quals_data(df11)
quals_21 = process_quals_data(df21)
occ_11 = process_occ_data(df11)
occ_21 = process_occ_data(df21)
nssec11 = process_nssec_data(df11)
nssec21 = process_nssec_data(df21)

# %%
df_11 = pd.concat([df11['Median_hp'], df11['Median_inc'], quals_11, occ_11], axis=1)
df_21 = pd.concat([df21['Median_hp'], df21['Median_inc'], quals_21, occ_21], axis=1)

df_11.columns = ['House Prices', 'Median Income', 'Share of Level 4+ Qualifications',
                 'Share of Knowledge Workers']
df_21.columns = ['House Prices', 'Median Income', 'Share of Level 4+ Qualifications',
                 'Share of Knowledge Workers']

df_11.to_csv(os.path.join(analytical, 'Untransformed-Data-2011.csv.gz'), compression='gzip', index=True)
df_21.to_csv(os.path.join(analytical, 'Untransformed-Data-2021.csv.gz'), compression='gzip', index=True)
# %%
#  Trials
# 0.78224765 ['Higher managerial','Lower managerial','Small employers']
# 0.79965654 ['Higher managerial', 'Lower managerial']
ns_11 = pd.concat([hp11, quals_11, occ_11, nssec11], axis=1)
ns_21 = pd.concat([hp21, quals_21, occ_21, nssec21], axis=1)

# 0.71570178
ns_11 = pd.concat([hp11, hinc11, quals_11, occ_11, nssec11], axis=1)
ns_21 = pd.concat([hp21, hinc21, quals_21, occ_21, nssec21], axis=1)

# 0.74435815
ns_11 = pd.concat([hp11, hinc11, quals_11, occ_11], axis=1)
ns_21 = pd.concat([hp21, hinc11, quals_21, occ_21], axis=1)

# 0.69399029
ns_11 = pd.concat([hp11, hinc11, hpr11, quals_11, occ_11, nssec11], axis=1)
ns_21 = pd.concat([hp21, hinc21, hpr21, quals_21, occ_21, nssec21], axis=1)

ns_11 = pd.concat([hp11, hinc11, hpr11, quals_11, occ_11], axis=1)
ns_21 = pd.concat([hp21, hinc21, hpr21, quals_21, occ_21], axis=1)

ns_11 = pd.concat([hp11, hinc11, quals_11, occ_11], axis=1)
ns_21 = pd.concat([hp21, hinc21, quals_21, occ_21], axis=1)

ns_11.columns = ['House Prices 2011', 'Share of Level 4+ Qualifications 2011',
                 'Share of Knowledge Workers 2011', 'Share of High NS-Sec 2011']
ns_21.columns = ['House Prices 2021', 'Share of Level 4+ Qualifications 2021',
                 'Share of Knowledge Workers 2021', 'Share of High NS-Sec 2021']

ns_11 = pd.concat([df11['Median_hp'], df11['Median_inc'], quals_11, occ_11], axis=1)
ns_21 = pd.concat([df21['Median_hp'], df21['Median_inc'], quals_21, occ_21], axis=1)

ns_11 = pd.concat([hp11, df11['Median_inc'], quals_11, occ_11, nssec11], axis=1)
ns_21 = pd.concat([hp21, df21['Median_inc'], quals_21, occ_21, nssec21], axis=1)

ns_11 = pd.concat([hp11_box, hinc11_box, hpr11_box, quals_11, occ_11, nssec11], axis=1)
ns_21 = pd.concat([hp21_box, hinc21_box, hpr21_box, quals_21, occ_21, nssec21], axis=1)

# %% Final
ns_11 = pd.concat([hp11, hinc11, quals_11, occ_11], axis=1)
ns_21 = pd.concat([hp21, hinc21, quals_21, occ_21], axis=1)

ns_11.columns = ['Log of House Prices 2011', 'Log of Median Income 2011', 'Share of Level 4+ Qualifications 2011',
                 'Share of Knowledge Workers 2011']
ns_21.columns = ['Log of House Prices 2021', 'Log of Median Income 2021', 'Share of Level 4+ Qualifications 2021',
                 'Share of Knowledge Workers 2021']

# Create dataset of indicator data
X_11 = ns_11.values
X_21 = ns_21.values

#  Join 2011 and 2021 datasets and sanity-check
score_inds = np.concatenate((X_11, X_21), axis=0)

print("Any infinite values? " + str(~np.isfinite(score_inds).any()))
print("Any NaN values? " + str(np.isnan(score_inds).any()))

#  Median removal and Unit scaling
scaler = preprocessing.RobustScaler()
scaler.fit(score_inds)
score_inds = scaler.transform(score_inds)

print("Data scaled and transformed.")

pca_full = decomposition.PCA()  # Use all Principal Components
pca_full.fit(score_inds)  # Train model on data
score_full_T = pd.DataFrame(pca_full.transform(score_inds))  # Transform data using model

print("The amount of explained variance of the  score using each component is...")
print(pca_full.explained_variance_ratio_)
# %%
# Adapted from https://stackoverflow.com/questions/22984335/recovering-features-names-of-explained-variance-ratio-in-pca-with-sklearn
i = np.identity(score_inds.shape[1])  # identity matrix

coef = pca_full.transform(i)

loadings = pd.DataFrame(coef, index=ns_11.columns)
loadings.to_csv(os.path.join(analytical, 'Loadings-2011.csv.gz'), compression='gzip', index=True)
loadings

# %%
#  Fitting PCA Model to derive  score
pca = decomposition.PCA(n_components=1)  # Only need 1st Principal Component
pca.fit(score_inds)  # Train model on data
score_inds_T = pd.DataFrame(pca.transform(score_inds))  # Transform data using model

print("The amount of explained variance of the  score is: {0:6.5f}".format(pca.explained_variance_ratio_[0]))
i = np.identity(score_inds.shape[1])  # identity matrix

coef = pca.transform(i)

loadings = pd.DataFrame(coef, index=ns_11.columns)
loadings
# %%
#  Split transformed data into 2011 and 2021 datasets
#  Note the way we do this to deal with missing data (if any)
status_11 = score_inds_T.loc[0:len(X_11) - 1, 0]
status_21 = score_inds_T.loc[len(X_11):, 0]

# Create dfs from the two sets of scores
ns_11 = ns_11.assign(status=pd.Series(status_11).values)
ns_21 = ns_21.assign(status=pd.Series(status_21).values)

# Join them together so we've got a single df for 2011 and 2021
res = ns_11.merge(ns_21, how='outer', suffixes=('_11', '_21'), left_index=True, right_index=True)

# Rename columns for consistency with Jordan's code
res.rename(columns={'status_11': 'score_11', 'status_21': 'score_21'}, inplace=True)

# Sanity check
res.head(3)

#  Compute rank of LSOA in 2001 (so low rank = 'low status')
res['RANK_11'] = res.score_11.rank(ascending=False)

#  Compute rank of LSOA in 2021 (so low rank = 'low status')
res['RANK_21'] = res.score_21.rank(ascending=False)

#  Compute amount by which LSOA has ascended (so +ve = status improvement; -ve = status decline)
res.loc[:, 'score_ASC'] = res.loc[:, 'score_21'] - res.loc[:, 'score_11']

import re

#  Calculate LSOA percentile score in 01
res.loc[:, 'score_PR_11'] = res.RANK_11.rank(ascending=False, pct=True) * 100

#  Calculate LSOA percentile score in 21
res.loc[:, 'score_PR_21'] = res.RANK_21.rank(ascending=False, pct=True) * 100

#  Calculate percentile change (so +ve = 'moved up' in the world; -ve = 'moved down')
res.loc[:, 'score_PR_ASC'] = res.loc[:, 'score_PR_21'] - res.loc[:, 'score_PR_11']
# res.loc[:,'score_PR_ASC'] = res.loc[:,'score_PR_11']-res.loc[:,'score_PR_21']

inp = res.loc[:, [x for x in res.columns if 'score' not in x and 'RANK' not in x]]

# Tidy up the naming
inp.rename(columns=lambda x: re.sub('_21', ' 2021', re.sub('_11', ' 2011', x)), inplace=True)
inp.rename(columns=lambda x: re.sub('kw_pct', 'Knowledge Worker Percentage', x), inplace=True)
inp.rename(columns=lambda x: re.sub('he_pct', 'Highly-Educated Percentage', x), inplace=True)
inp.rename(columns=lambda x: re.sub('hs_pct', 'Higher socio-economic Percentage', x), inplace=True)
inp.rename(columns=lambda x: re.sub('hp', 'Property Prices (Logged)', x), inplace=True)
inp.rename(columns=lambda x: re.sub('hinc', 'Household Income (Logged)', x), inplace=True)
inp.rename(columns=lambda x: re.sub('hpr', 'Median Rent (Logged)', x), inplace=True)

# Save to file (note that we are also saving some info about the input variables as we use these as well)
res[['RANK_11', 'RANK_21', 'score_11', 'score_21', 'score_ASC', 'score_PR_11', 'score_PR_21', 'score_PR_ASC']].to_csv(
    os.path.join(analytical, 'Scores.csv.gz'), compression='gzip', index=True)
inp[[x for x in inp.columns if '2011' in x]].to_csv(os.path.join(analytical, 'Inputs-2011.csv.gz'), compression='gzip',
                                                    index=True)
inp[[x for x in inp.columns if '2021' in x]].to_csv(os.path.join(analytical, 'Inputs-2021.csv.gz'), compression='gzip',
                                                    index=True)
print('Done')
# %%
#  Sanity check
res[['score_11', 'score_21', 'RANK_11', 'RANK_21', 'score_PR_11', 'score_PR_21', 'score_PR_ASC']].sample(5, random_state=r_state)

# The lowest-ranked (highest status) LSOAs
res.loc[res['RANK_11'] < 5, :].sort_values('RANK_11')
# %%
# The highest-ranked (lowest status) LSOAs
res.loc[res['RANK_11'] > (res.RANK_11.max() - 5), :].sort_values('RANK_11')
# %%
# Biggest falls in percentile status
res.sort_values('score_PR_ASC').head(5)
# %%
# Biggest gains in percentile status
res.sort_values('score_PR_ASC', ascending=False).head(5)
# %%
sns.jointplot(x='score_11', y='score_21', data=res, kind='scatter', s=3, color='k', height=7, ratio=5, space=0, linewidth=1)
plt.show()
# %%
sns.jointplot(x='score_PR_11', y='score_PR_21', data=res, kind='scatter', s=3, color='k')
plt.show()
# %%
sns.jointplot(x='RANK_11', y='RANK_21', data=res, kind='scatter', s=3, color='k')
plt.show()

# %%
sns.scatterplot(x='score_ASC', y='Median_hp_11', data=res)
plt.show()
# %%
sources = {
    'demographics': {
        'age_struct': 'Age Structure',
        'ethnic': 'Ethnicity',
        'hh_comp': 'Household Composition',
        'density': 'Density',
        'marital_status': 'Marital Status',
    },

    'housing': {
        'acc_type': 'Accommodation Type',
        'sales_num': 'Number-of-Sales',
        'hsng_tnr': 'Tenure',
        'period': 'Dwelling-Period',
        'cv': 'Cars and Vans',
        'rent': 'Private-rents-LSOAs'
        # 'income'     :'Private-Income-LSOAs'
    },
    'work': {
        'hrs_wrkd': 'Hours Worked',
        'industry': 'Industry',
        'active': 'Economic Activity',
        # 'occ'        :'Occupation', # Included in next notebook
        'quals': 'Qualifications',  # Included in next notebook
        'ns_sec': 'ns_sec'
    },
    'social': {
        'green_acc_data': 'Greenspace',
        'fare_zone': 'Fare Zone',
        'travel_time': 'Travel Time To Bank'
        # 'crime'     :'Crime'
    }
}

# Not converted to percentages'
not_to_pct = [
    'density',
    'add_dwellings',
    'green_acc_data',
    'rent',
    'income',
    'sales_num',
    'fare_zone',
    'travel_time'
]

# Convert raw data in datasets to percentages
def to_share(df):
    df.rename(columns={'TOTAL': 'total', 'Total': 'total'}, inplace=True)

    if 'total' not in df.columns.values:
        df.loc[:, 'total'] = df.sum(axis=1)  # Total number of people

    pred_data = pd.DataFrame()

    for n in df.columns.tolist():
        # print("Converting " + n)
        pred_data.loc[:, n] = (df.loc[:, n] / df.loc[:, 'total'])

    pred_data.drop(['total'], axis=1, inplace=True)

    pred_data.describe()

    return pred_data


# Load data sets for each Census year based on schema above
def load_data(year, template):
    datasets = {}
    for group in template.keys():
        print("Dataset group: " + group)
        for ds in template[group].keys():
            print("\tLoading dataset: " + ds)

            # Tentative path
            if template[group][ds] in ['Crime']:
                tpath = os.path.join('data', 'factors', group, template[group][ds] + '-Converted')
            elif year == 2021:
                if template[group][ds] in ['Dwelling-Period', 'Greenspace', 'Number-of-Sales', 'Fare Zone',
                                           'Travel Time To Bank']:
                    tpath = os.path.join('data', 'factors', group, template[group][ds])
                else:
                    tpath = os.path.join('data', 'factors', group, template[group][ds] + '-Converted')
            else:
                tpath = os.path.join('data', 'factors', group, template[group][ds])

            # Load the data set
            if os.path.isfile("-".join([tpath, str(year)]) + ".csv"):
                print("\t\tFound: " + "-".join([tpath, str(year)]) + ".csv")
                datasets[ds] = pd.read_csv("-".join([tpath, str(year)]) + ".csv")

            elif os.path.isfile(tpath + ".csv"):
                print("\t\tFound: " + tpath + ".csv")
                datasets[ds] = pd.read_csv(tpath + ".csv")

            else:
                print("==> Couldn't find data for: " + template[group][ds] + " <==")
                print("Tried: " + "-".join([tpath, str(year)]) + "; " + tpath + ".csv")

            if datasets[ds].index.name != 'lsoacd':
                datasets[ds].set_index('lsoacd', inplace=True)  # predictor variables only

            if ds not in not_to_pct:
                datasets[ds] = to_share(datasets[ds])
            else:
                print("\t\tNot converting to percent.")

    return datasets


print("Loading 2011 datasets...")
datasets_11 = load_data(2011, sources)

print("Loading 2021 datasets...")
datasets_21 = load_data(2021, sources)

sets = {
    '11': datasets_11,
    '21': datasets_21
}

print("Done.")

# %%
main_datasets_dict = dict()

for year, dataset in sets.items():
    #  Create combined dataset
    main_dataset = pd.DataFrame(index=sets['11']['acc_type'].index)  # Initialise the df
    for key, value in iter(sorted(dataset.items())):
        print("Merging " + key + " on to dataset.")
        main_dataset = main_dataset.merge(value, left_index=True, right_index=True, how='left')

    main_datasets_dict[year] = main_dataset

    #  Check for missing values
    print("Missing values (if any) to be filled:")
    for c in main_dataset.columns[main_dataset.isnull().any()]:
        print("\t" + c + " has " + str(main_dataset[c].isnull().sum()) + " null values.")
        print("\t\t" + ", ".join(main_dataset[main_dataset[c].isnull()].index.values))

    main_dataset.fillna(0, inplace=True)

    print("Main dataset built for 20" + year + ".")
    print(" ")

main_datasets_dict['11'].rename(columns=lambda x: re.sub(' 2011$', '', x), inplace=True)
main_datasets_dict['21'].rename(columns=lambda x: re.sub(' 2021$', '', x), inplace=True)

print("2011 Shape: " + str(main_datasets_dict['11'].shape))
print("2021 Shape: " + str(main_datasets_dict['21'].shape))

# %%
s21 = set(main_datasets_dict['21'].columns)
s11 = set(main_datasets_dict['11'].columns)
print("2021 variables diff'd against 2021 variables: " + str(s21.difference(s21)))
print("2011 variables diff'd against 2011 variables: " + str(s11.difference(s11)))
# %%
main_datasets_dict['11'].sample(3, random_state=r_state)
# %%
for key, value in main_datasets_dict.items():
    print("Saving data for 20" + key)
    value.to_csv(os.path.join(analytical, 'Predictor-20' + key + '-Data.csv.gz'), compression='gzip', index=True)

print("Done.")

df_data_2011, df_data_2021 = main_datasets_dict['11'], main_datasets_dict['21']
#
for col in df_data_2011:
    difference = df_data_2011[col] - df_data_2021[col]
    if np.mean(difference) == 0:
        print(col, np.mean(difference))
