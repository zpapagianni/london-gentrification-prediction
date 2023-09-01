import random
import warnings

import matplotlib.pyplot as plt
import numpy as np

import os
import pandas as pd
import seaborn as sns
import re

from sklearn import preprocessing
warnings.filterwarnings("ignore")
r_state = 42
random.seed(r_state)
np.random.seed(r_state)
# %%
# Create folders path
lkp = os.path.join('data', 'lkp')
shp = os.path.join('data', 'shp')
src = os.path.join('data', 'src')
analytical = os.path.join('data', 'analytical')

# %%
score = pd.read_csv(os.path.join(analytical, 'Scores.csv.gz'), index_col=0)  # score scores

# Scores
score.drop(['RANK_11', 'RANK_21'], axis=1, inplace=True)
score.rename(columns={
    'score_11': 'score 2111',
    'score_21': 'score 2021',
    'score_ASC': 'score Ascent 2111-2021',
    'score_PR_11': 'score 2111 Percentile',  # 99 = High-status
    'score_PR_21': 'score 2021 Percentile',  # 99 = High-status
    'score_PR_ASC': 'score Percentile Ascent 2111-2021'
}, inplace=True)

d11input = pd.read_csv(os.path.join(analytical, 'Inputs-2011.csv.gz'), index_col=0)  # score inputs
d21input = pd.read_csv(os.path.join(analytical, 'Inputs-2021.csv.gz'), index_col=0)  # score inputs

# Rename to remove confusion
d11input.rename(columns=lambda x: re.sub(' 2011', '', x), inplace=True)
d21input.rename(columns=lambda x: re.sub(' 2021', '', x), inplace=True)
# %%
#  Read in processed datasets
d11 = pd.read_csv(os.path.join(analytical, 'Predictor-2011-Data.csv.gz'), compression='gzip',
                  index_col=0)  # Main dataset for 2111
d21 = pd.read_csv(os.path.join(analytical, 'Predictor-2021-Data.csv.gz'), compression='gzip',
                  index_col=0)  # Main dataset for 2021

d11 = pd.merge(d11input, d11, how='inner', left_index=True, right_index=True)
d21 = pd.merge(d21input, d21, how='inner', left_index=True, right_index=True)
# %%
# Sanity check
if d11.shape[0] != 4835:
    print("Wrong number of rows in d11: " + d11.shape[0])
if d21.shape[0] != 4835:
    print("Wrong number of rows in d21: " + d21.shape[0])

print("Have " + str(len(d11.columns) + 1) + " variables to work with.")
d11.sample(3, random_state=r_state)

# Sanity check
s11 = set(d11.columns)
s21 = set(d21.columns)
print("2011 vs 2021 variable check: " + str(s11.difference(s21)))
print("2021 vs 2011 variable check: " + str(s21.difference(s11)))

# %%
score.describe()
descriptives = pd.DataFrame()
for c in d11.columns:
    descriptives = descriptives.append(pd.concat([d11[c].describe(), d21[c].describe()], axis=0, ignore_index=True),
                                       ignore_index=False)

descriptives.columns = ['2011 Count', '2011 Mean', '2011 StD', '2011 Min', '2011 LQ', '2011 Median', '2011 UQ',
                        '2011 Max',
                        '2021 Count', '2021 Mean', '2021 StD', '2021 Min', '2021 LQ', '2021 Median', '2021 UQ',
                        '2021 Max']

# This enables to re-use the same sample below
dsample = descriptives.sample(4, random_state=r_state).index.values
dsample = np.append(dsample,
                    ['Percentage with Level 4+ Qualifications', 'Percentage of Knowledge Workers'])

# %%
rs1 = preprocessing.RobustScaler(with_centering=False, quantile_range=(25.0, 75.0))

standard_scaler = preprocessing.StandardScaler()

#  Train on 2001 data set
rs1.fit(d11)

# Apply the same unit variance scaling to both years
d11_trs1 = pd.DataFrame(data=rs1.transform(d11), index=d11.index, columns=d11.columns)
d21_trs1 = pd.DataFrame(data=rs1.transform(d21), index=d21.index, columns=d21.columns)

# Create new robust scaler for centering
# _without_ common scaling.
rs2 = preprocessing.RobustScaler(with_scaling=False)

# Centre independently
d11_trs2 = pd.DataFrame(data=rs2.fit_transform(d11_trs1), index=d11.index, columns=d11.columns)
d21_trs2 = pd.DataFrame(data=rs2.fit_transform(d21_trs1), index=d21.index, columns=d21.columns)

#  Write the transformed data to csv
d11_trs2.to_csv(os.path.join(analytical, '2011-Data-Transformed_and_Scaled.csv.gz'), compression='gzip', index=True)
d21_trs2.to_csv(os.path.join(analytical, '2021-Data-Transformed_and_Scaled.csv.gz'), compression='gzip', index=True)

# standard_scaler =preprocessing.StandardScaler()
# standard_scaler.fit(d11)
#
# # Apply the same unit variance scaling to both years
# d11_trs1 = pd.DataFrame(data=rs1.transform(d11), index=d11.index, columns=d11.columns)
# d21_trs1 = pd.DataFrame(data=rs1.transform(d21), index=d21.index, columns=d21.columns)
#
# d11_trs1.to_csv(os.path.join(analytical,'2011-Data-Transformed_and_St-Scaled.csv.gz'), compression='gzip', index=True)
# d21_trs1.to_csv(os.path.join(analytical,'2021-Data-Transformed_and_St-Scaled.csv.gz'), compression='gzip', index=True)
#
# print("Done.")
# %%
descriptives_trs1 = pd.DataFrame()
for c in d11_trs1.columns:
    descriptives_trs1 = descriptives_trs1.append(
        pd.concat([d11_trs1[c].describe(), d21_trs1[c].describe()], axis=0, ignore_index=True), ignore_index=False)

descriptives_trs1.columns = ['2011 Count', '2011 Mean', '2011 StD', '2011 Min', '2011 LQ', '2011 Median', '2011 UQ',
                             '2011 Max',
                             '2021 Count', '2021 Mean', '2021 StD', '2021 Min', '2021 LQ', '2021 Median', '2021 UQ',
                             '2021 Max']

# Useful, but time-consuming
# plot_checks(d11_trs1, dsample, 'First-transform')

descriptives_trs1[descriptives_trs1.index.isin(dsample)][
    ['2011 Min', '2021 Min', '2011 Max', '2021 Max', '2011 Median', '2021 Median', '2011 Mean', '2021 Mean']
]
