import matplotlib.pyplot as plt
# For reproducibility
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shap

from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor

pd.set_option('display.max_columns', None)
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_regression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit

r_state = 42
random.seed(r_state)
np.random.seed(r_state)

# %%
analytical = os.path.join('data', 'analytical')
output = os.path.join('data', 'output')
shp = os.path.join('data', 'shp')

if os.path.exists(output):
    shutil.rmtree(output)


def load_status_scores():
    status = pd.read_csv(os.path.join(analytical, 'Scores.csv.gz'), index_col=0)  # score scores

    # Scores
    status.drop(['RANK_11', 'RANK_21'], axis=1, inplace=True)
    status.rename(columns={'Score_11': 'Score 2011', 'Score_21': 'Score 2021', 'Score_ASC': 'Score Ascent 2011-2021',
                           'Score_PR_11': 'Score 2011 Percentile',  # 99 = High-status
                           'Score_PR_21': 'Score 2021 Percentile',  # 99 = High-status
                           'Score_PR_ASC': 'Score Percentile Ascent 2011-2021'}, inplace=True)
    return status


def model_report(model, y_true):
    y_hat = model.predict(X_test)

    txt = ''

    # Regression metrics,
    txt += "R2:        {0:8.5f}".format(metrics.r2_score(y_true, y_hat)) + "\n"  # R2 - Coefficient of determination
    txt += "MSE:       {0:8.5f}".format(
        metrics.mean_squared_error(y_true, y_hat)) + "\n"  # Mean squared error regression loss
    txt += "MAE:       {0:8.5f}".format(
        metrics.mean_absolute_error(y_true, y_hat)) + "\n"  # Mean absolute error regression loss
    txt += "Expl. Var: {0:8.5f}".format(
        metrics.explained_variance_score(y_true, y_hat)) + "\n"  # Explained variance regression score function
    txt += "\n"
    return txt


# %%
score = load_status_scores()  # score scores in 2021

#  Read the transformed data
d11_trs2 = pd.read_csv(os.path.join(analytical, '2011-Data-Transformed_and_Scaled.csv.gz'), index_col=0)
d21_trs2 = pd.read_csv(os.path.join(analytical, '2021-Data-Transformed_and_Scaled.csv.gz'), index_col=0)

# Data about variables used later in process
vardb = pd.read_csv(os.path.join('data', 'variables.csv'), index_col=False)
vardb.drop('Description', axis=1, inplace=True)

X_full = d11_trs2

X_full.drop(['Level 4 qualifications and above', 'Unemployed_x'], axis=1, inplace=True)
d21_trs2.drop(['Level 4 qualifications and above', 'Unemployed_x'], axis=1, inplace=True)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(d11_trs2, score['Score Ascent 2011-2021'], test_size=0.2, random_state=r_state)
# X_train, X_test, y_train, y_test = model_selection.train_test_split(d11_trs2, score['Score 2021'], test_size=0.2, random_state=r_state)


# plt.scatter(score['Score 2021'],d11_trs2['House Prices'])
# plt.show()

# %%
geo_data = pd.read_csv('data/shp/LSOAs Coordinates 2011.cvs')
geo_data.set_index('LSOA11CD', inplace=True)
coordinates = (geo_data['x_degrees'], geo_data['y_degrees'])

# Calculate the diagonal distance
x_max, x_min = geo_data['x_degrees'].max(), geo_data['x_degrees'].min()
y_max, y_min = geo_data['y_degrees'].max(), geo_data['y_degrees'].min()
diagonal_distance = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
# Calculate spacing to have 5 blocks (you can change num_blocks if needed)
num_blocks = 3
spacing = diagonal_distance / num_blocks


# %%
# Define a function to create spatial blocks
def create_spatial_blocks(geodata, num_blocks):
    # Calculate block IDs based on x and y coordinates
    x_bin = ((geodata['x_degrees'] - x_min) / spacing).astype(int)
    y_bin = ((geodata['y_degrees'] - y_min) / spacing).astype(int)
    block_ids = x_bin + y_bin * num_blocks
    return block_ids


# Create spatial blocks
block_ids = create_spatial_blocks(geo_data, num_blocks)


# %%
# Define function for spatial train-test split
def spatial_train_test_split(X, y, block_ids, test_size=0.2, random_state=None):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(splitter.split(X, block_ids))
    train_X, test_X = X.iloc[train_indices], X.iloc[test_indices]
    train_y, test_y = y.iloc[train_indices], y.iloc[test_indices]
    block_ids_train = block_ids[train_indices]
    return train_X, test_X, train_y, test_y, block_ids_train


# Perform spatially coherent train-test split
X_train, X_test, y_train, y_test, block_ids_train = spatial_train_test_split(X_full, score['Score 2021'], block_ids,
                                                                             test_size=0.2, random_state=42)

# %% Linear Regression
lr_model = linear_model.LinearRegression(fit_intercept=True, copy_X=True)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print(model_report(lr_model, y_test))

# %% Random Forest Untuned
rf_untuned = RandomForestRegressor(random_state=42)
rf_untuned.fit(X_train, y_train)
y_pred = rf_untuned.predict(X_test)
print(model_report(rf_untuned, y_test))

# %% Random Forest Tuned

# Define hyperparameter grid for ExtraTreesRegressor
param_grid = {'max_depth': [None, 2, 5, 10, 15, 20], 'max_features': [None, 2, 3, 5, 10],
              'min_samples_leaf': [1, 2, 3, 4], 'n_estimators': [50, 100, 150, 200, 300]}

# Initialize ExtraTreesRegressor
rf_model = ExtraTreesRegressor(n_jobs=-1, random_state=42)
sss = GroupKFold(n_splits=5)
# Initialize GridSearchCV for hyperparameter tuning with spatial cross-validation
grid_search = GridSearchCV(rf_model, param_grid, cv=sss, scoring='neg_mean_squared_error')

# Fit GridSearchCV to training data
grid_search.fit(X_train, y_train, groups=block_ids_train)

# Print the best hyperparameters and corresponding score
print('Done')
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
best_rf = grid_search.best_estimator_
print(model_report(best_rf, y_test))
y_pred_rf = best_rf.predict(X_test)
# %%
best_rf_params = {'max_depth': 20, 'max_features': None, 'min_samples_leaf': 2, 'n_estimators': 300}

best_rf = ExtraTreesRegressor(**best_rf_params, random_state=42)
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
print(model_report(best_rf, y_test))

# %% Fit vs Predicted

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# %% Shapley Values
explainer = shap.Explainer(best_rf)
shap_values = explainer(X_train)

shap.summary_plot(shap_values, X_train, plot_size=[15, 8], show=False)
shap.summary_plot(shap_values, X_train, plot_size=[20, 12], show=False)
plt.savefig('plots/shap_score_rf.png', bbox_inches='tight', dpi=100)
# %% Gradient Boosting Machines Untuned
gb_model_untuned = GradientBoostingRegressor()
gb_model_untuned.fit(X_train, y_train)
y_pred = gb_model_untuned.predict(X_test)
print(model_report(gb_model_untuned, y_test))
# %%

# Create the parameter grid based on the results of random search
param_grid_gb = {'n_estimators': [50, 100, 200, 300],  # Number of boosting stages (trees)
                 'learning_rate': [0.01, 0.1, 0.2],  # Learning rate for each boosting stage
                 'max_depth': [None, 3, 5, 7], 'min_samples_leaf': [1, 2, 4],
                 # Minimum number of samples required to be at a leaf node
                 'max_features': [None, 5, 10], }

# Create a based model
gb_model = GradientBoostingRegressor(random_state=42)

# Instantiate the grid search model
sss = GroupKFold(n_splits=5)
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid_gb, cv=sss, verbose=5)

grid_search.fit(X_train, y_train, groups=block_ids_train)

best_params = grid_search.best_params_
best_gb_model = grid_search.best_estimator_
print(model_report(best_gb_model, y_test))
y_pred_gb = best_gb_model.predict(X_test)
# %%
best_gb_params = {'learning_rate': 0.1, 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 4,
                  'n_estimators': 300}

best_gb_model = GradientBoostingRegressor(**best_gb_params, random_state=42)
best_gb_model.fit(X_train, y_train)
y_pred = best_gb_model.predict(X_test)
print(model_report(best_gb_model, y_test))

# %%y_pred= best_grid.predict(X_test)
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_gb)
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# %% Shapley Values

explainer = shap.Explainer(best_gb_model)
shap_values_gb = explainer(X_train)

shap.summary_plot(shap_values_gb, X_train, plot_size=[20, 12])
shap.summary_plot(shap_values_gb, X_train, plot_size=[20, 12], show=False)
plt.savefig('plots/shap_score_gb.png', bbox_inches='tight', dpi=100)

# shap.summary_plot(shap_values, X_train, plot_type="bar")

# %%
y_pred_rf_full = best_rf.predict(X_full)
predicted21_rf = pd.DataFrame({'lsoacd': pd.Series(d11_trs2.index),
                               'Score 2021 (Predicted)': pd.Series(y_pred_rf_full)})  # Combine with list of areas
predicted21_rf.set_index('lsoacd', inplace=True)
predicted21_rf.sample(3, random_state=r_state)
# %%
predicted21 = predicted21_rf.merge(score, left_index=True, right_index=True, how='inner')

predicted21['Score Ascent 2011-2021 (Predicted)'] = predicted21.loc[:, 'Score 2021 (Predicted)'] - predicted21.loc[:,
                                                                                                   'Score 2011']
predicted21['Score Divergence'] = predicted21.loc[:, 'Score 2021 (Predicted)'] - predicted21.loc[:, 'Score 2021']
predicted21['Ascent Divergence'] = predicted21.loc[:, 'Score Ascent 2011-2021 (Predicted)'] - predicted21.loc[:,
                                                                                              'Score Ascent 2011-2021']

predicted21.sort_index(axis=1, inplace=True)

predicted21.to_csv(os.path.join(analytical, 'RF Predicted 2021.csv.gz'), compression='gzip', index=True)

# Sanity check
print("Results data frame has " + str(predicted21.shape[0]) + " rows.")
predicted21.sample(5, random_state=r_state)

# %%
fig = plt.figure()
sns.distplot(predicted21['Score Divergence'], kde=True)
plt.title('Gentrification Score Divergence')
plt.xlabel("")
plt.savefig('plots/score_divergence_rf.png', bbox_inches='tight', dpi=100)
plt.show()
print("Done.")

# %%
fig = plt.figure()
sns.scatterplot(x='Score Ascent 2011-2021', y='Score Ascent 2011-2021 (Predicted)', data=predicted21)
plt.plot([-0.5, 3], [-0.5, 3], color='blue', linewidth=2)
plt.title('Predicted vs True Gentrification Score Change')
plt.xlabel("True Gentr. Score Change")
plt.ylabel("Predicted Gentr. Score Change")
plt.savefig('plots/gentr_change_rf.png', bbox_inches='tight', dpi=100)
plt.show()
print("Done.")

# %% Predict
y_pred_gb_full = best_gb_model.predict(X_full)
predicted21_gb = pd.DataFrame({'lsoacd': pd.Series(d11_trs2.index),
                               'Score 2021 (Predicted)': pd.Series(y_pred_gb_full)})  # Combine with list of areas
predicted21_gb.set_index('lsoacd', inplace=True)
predicted21_gb.sample(3, random_state=r_state)
# %%
predicted21_gb = predicted21_gb.merge(score, left_index=True, right_index=True, how='inner')

predicted21_gb['Score Ascent 2011-2021 (Predicted)'] = predicted21_gb.loc[:,
                                                       'Score 2021 (Predicted)'] - predicted21_gb.loc[
                                                                                   :, 'Score 2011']

predicted21_gb['Score Divergence'] = predicted21_gb.loc[:, 'Score 2021 (Predicted)'] - predicted21_gb.loc[:,
                                                                                       'Score 2021']
predicted21_gb['Ascent Divergence'] = predicted21_gb.loc[:, 'Score Ascent 2011-2021 (Predicted)'] - predicted21_gb.loc[
                                                                                                    :,
                                                                                                    'Score Ascent 2011-2021']

predicted21_gb.sort_index(axis=1, inplace=True)

predicted21_gb.to_csv(os.path.join(analytical, 'GB Predicted 2021.csv.gz'), compression='gzip', index=True)

# Sanity check
print("Results data frame has " + str(predicted21.shape[0]) + " rows.")
predicted21.sample(5, random_state=r_state)
# %%
fig = plt.figure()
sns.distplot(predicted21_gb['Score Divergence'], kde=True)
plt.title('Gentrification Score Divergence')
plt.xlabel("")
plt.savefig('plots/score_divergence_gb.png', bbox_inches='tight', dpi=100)
plt.show()
print("Done.")

# %%
fig = plt.figure()
sns.scatterplot(x='Score Ascent 2011-2021', y='Score Ascent 2011-2021 (Predicted)', data=predicted21_gb)
plt.plot([-0.5, 3], [-0.5, 3], color='blue', linewidth=2)
plt.title('Predicted vs True Gentrification Score Change')
plt.xlabel("True Gentr. Score Change")
plt.ylabel("Predicted Gentr. Score Change")
plt.savefig('plots/gentr_change_gb.png', bbox_inches='tight', dpi=100)
plt.show()
print("Done.")

# %%
cols = ['Score 2011', 'Score 2021 (Predicted)', 'Score 2021', 'Score 2031 (Predicted)', 'Score 2011 Percentile',
        'Score 2021 Percentile', 'Score 2031 Percentile', 'Score Ascent 2011-2021',
        'Score Ascent 2011-2021 (Predicted)',
        'Score Ascent 2021-2031 (Predicted)', 'Score Percentile Ascent 2011-2021', 'Score Percentile Ascent 2021-2031',
        'Score Divergence', 'Ascent Divergence']

#  Make future predictions
y_pred_31 = best_gb_model.predict(d21_trs2)  # Make predictions using data from 2021
predicted31_gb = pd.DataFrame({'lsoacd': pd.Series(d21_trs2.index), 'Score 2031 (Predicted)': pd.Series(y_pred_31)})
predicted31_gb.set_index('lsoacd', inplace=True)

predicted31_gb.to_csv(os.path.join(analytical, 'GB-Predicted Scores 2031.csv.gz'), compression='gzip',
                      index=True)  # Write results to csv
predicted31_gb.sample(3, random_state=r_state)

pdf_gb = predicted21_gb.merge(predicted31_gb, left_index=True, right_index=True,
                              how='left')  # Integrate score 2021 predictions into score score data
pdf_gb.loc[:, 'Score Ascent 2021-2031 (Predicted)'] = pdf_gb['Score 2031 (Predicted)'] - pdf_gb.loc[:,
                                                                                         'Score 2021']  # Compute score score in 2021

#  Compute rank in 2012
pdf_gb['Score 2031 Percentile'] = pdf_gb.loc[:, 'Score 2031 (Predicted)'].rank(ascending=True, pct=True) * 100

#  Compute change in LSOA ranking from 2011 to 2021
pdf_gb['Score Percentile Ascent 2021-2031'] = pdf_gb.loc[:, 'Score 2031 Percentile'] - pdf_gb.loc[:,
                                                                                       'Score 2021 Percentile']

pdf_gb[['Score 2011', 'Score 2021', 'Score 2031 (Predicted)', 'Score Ascent 2011-2021',
        'Score Ascent 2021-2031 (Predicted)']].sample(3, random_state=r_state)
pdf_gb = pdf_gb[cols]
pdf_gb.loc[:, fcols] = pdf_gb[fcols].astype(float).applymap('{0:.15f}'.format)
pdf_gb.to_csv(os.path.join(analytical, 'GB-Full Predictions.csv.gz'), compression='gzip', index=True)

# %%
#  Make future predictions
y_pred_31 = best_rf.predict(d21_trs2)  # Make predictions using data from 2021
predicted31_rf = pd.DataFrame({'lsoacd': pd.Series(d21_trs2.index), 'Score 2031 (Predicted)': pd.Series(y_pred_31)})
predicted31_rf.set_index('lsoacd', inplace=True)

predicted31_rf.to_csv(os.path.join(analytical, 'RF-Predicted Scores 2031.csv.gz'), compression='gzip',
                      index=True)  # Write results to csv
predicted31_rf.sample(3, random_state=r_state)

pdf_rf = predicted21.merge(predicted31_rf, left_index=True, right_index=True,
                           how='left')  # Integrate score 2021 predictions into score score data
pdf_rf.loc[:, 'Score Ascent 2021-2031 (Predicted)'] = pdf_rf['Score 2031 (Predicted)'] - pdf_rf.loc[:,
                                                                                         'Score 2021']  # Compute score score in 2021

#  Compute rank in 2012
pdf_rf['Score 2031 Percentile'] = pdf_rf.loc[:, 'Score 2031 (Predicted)'].rank(ascending=True, pct=True) * 100

#  Compute change in LSOA ranking from 2011 to 2021
pdf_rf['Score Percentile Ascent 2021-2031'] = pdf_rf.loc[:, 'Score 2031 Percentile'] - pdf_rf.loc[:,
                                                                                       'Score 2021 Percentile']

pdf_rf[['Score 2011', 'Score 2021', 'Score 2031 (Predicted)', 'Score Ascent 2011-2021',
        'Score Ascent 2021-2031 (Predicted)']].sample(3, random_state=r_state)
pdf_rf = pdf_rf[cols]
pdf_rf.loc[:, fcols] = pdf_rf[fcols].astype(float).applymap('{0:.15f}'.format)
pdf_rf.to_csv(os.path.join(analytical, 'RF-Full Predictions.csv.gz'), compression='gzip', index=True)


# %% Geographically weighted random forest

# Function to perform Geographically Weighted Random Forest
def geographically_weighted_random_forest(X, y, qw, fh, cds, wmethod):
    Outputs = []
    y_pred = pd.DataFrame(index=y.index, columns=['Prediction'])
    y_pred['Score 2021 (Predicted)'] = np.nan

    # Loop through each index in the feature matrix
    for i in X.index:
        neighbours = []
        ns = qw[cds.index(i)]

        # Extract neighbors based on weighting method
        for n in ns.keys():
            if wmethod == 'Queen':
                neighbours.append(fh[n][0][0])
            else:
                neighbours.append(fh[n.item()][0][0])

        # Select data for modeling from neighboring LSOAs
        X_obs = X.loc[neighbours]
        y_obs = y.loc[neighbours]

        # Create a Random Forest regressor
        RFi = RandomForestRegressor()
        RFi.fit(X_obs, y_obs)

        # Extract feature importances
        LVIi = RFi.feature_importances_
        LVIi[LVIi < 0] = 0  # Remove variables with negative scores

        # Check if number of variables in LVIi matches columns in X
        if len(LVIi) != len(X.columns):
            RFi.fit(X_obs, y_obs)
            LVIi = RFi.feature_importances_
            LVIi[LVIi < 0] = 0  # Remove variables with negative scores

        # Predict target variable for neighboring LSOAs
        YHATi = RFi.predict(X_obs)
        YHAT_trial = pd.DataFrame(YHATi)
        YHAT_trial.index = y_obs.index

        # Update predicted values with an average if available
        for cd in y_obs.index:
            y_pred_previous = y_pred.loc[cd]['Prediction']
            if np.isnan(y_pred_previous):
                y_pred.loc[cd] = YHAT_trial.loc[cd][0]
            else:
                y_pred.loc[cd] = (y_pred_previous + YHAT_trial.loc[cd][0]) / 2

        # Calculate performance metrics
        sc = metrics.r2_score(y_obs, YHATi, multioutput='variance_weighted')
        mse = metrics.mean_squared_error(y_obs, YHATi)
        mae = metrics.mean_absolute_error(y_obs, YHATi)
        var = metrics.explained_variance_score(y_obs, YHATi)
        ACCi = [sc, mse, mae, var]
        Outputs.append((LVIi, YHATi, ACCi))

    # Organize outputs into DataFrames
    LVI, YHAT, ACC = zip(*Outputs)
    LVI = pd.DataFrame(LVI)
    YHAT = pd.DataFrame(YHATi)
    ACC = pd.DataFrame(ACC)

    return LVI, YHAT, ACC, y_pred


# Load the spatial weights matrix from the shapefile
spatial_weights_file = os.path.join(shp, 'LSOA-2011-Weights.shp')
spatial_weights = weights.Queen.from_shapefile(spatial_weights_file)

# Load spatial weights using a kernel function from a shapefile
qw = ps.lib.weights.Kernel.from_shapefile(os.path.join(shp, 'LSOA-2011-Weights.shp'))
fh = ps.lib.io.open(os.path.join('data', 'shp', 'LSOA-2011-Weights.dbf'))
cds = fh.by_col['LSOA11CD']  # LSOA 2011 Census code
wmethod = 'Kernel'

# Load spatial weights using a Queen contiguity criterion from a shapefile
qw = ps.lib.weights.Queen.from_shapefile(os.path.join(shp, 'LSOA-2011-Weights.shp'))
fh = ps.lib.io.open(os.path.join('data', 'shp', 'LSOA-2011-Weights.dbf'))
cds = fh.by_col['LSOA11CD']  # LSOA 2011 Census code
wmethod = 'Queen'

LVI, YHAT, ACC, y_pred = geographically_weighted_random_forest(d11_trs2, score['Score 2021'], qw, fh, cds, wmethod)

np.mean(ACC)

# %% Save Results

predicted21_grf = pd.DataFrame(
    {'lsoacd': pd.Series(d11_trs2.index), 'Score 2021 (Predicted)': pd.Series(y_pred)})  # Combine with list of areas
predicted21_grf.set_index('lsoacd', inplace=True)
predicted21_grf.sample(3, random_state=r_state)

predicted21_grf = y_pred.merge(score, left_index=True, right_index=True, how='inner')
predicted21_grf.rename(columns={'Prediction': 'Score 2021 (Predicted)'}, inplace=True)

predicted21_grf['Score Ascent 2011-2021 (Predicted)'] = predicted21_grf.loc[:,
                                                        'Score 2021 (Predicted)'] - predicted21_grf.loc[:, 'Score 2011']

predicted21_grf['Score Divergence'] = predicted21_grf.loc[:, 'Score 2021 (Predicted)'] - predicted21_grf.loc[:,
                                                                                         'Score 2021']
predicted21_grf['Ascent Divergence'] = predicted21_grf.loc[:,
                                       'Score Ascent 2011-2021 (Predicted)'] - predicted21_grf.loc[
                                                                               :,
                                                                               'Score Ascent 2011-2021']

predicted21_grf.sort_index(axis=1, inplace=True)

predicted21_grf.to_csv(os.path.join(analytical, 'GRF-Predicted 2021.csv.gz'), compression='gzip', index=True)
