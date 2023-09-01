# Gentrification Prediction Project

This repository contains scripts and code for a gentrification prediction project using machine learning techniques. The project aims to predict gentrification patterns in various neighborhoods in London based on a range of socioeconomic indicators. Below is a brief overview of the scripts and their functions:
The input data can be found in the data zip file, which contains two folders: the 'zip' folder containing the shapefiles and the 'predictor' folder containing all the variables used in this project.

## Scripts:

### 00-Prepare Shapefiles.py
-  This script prepares the shapefiles for London's neighborhoods. It may involve cleaning, formatting, or merging shapefiles for further analysis.

### 01-Prepare-Predictors.py
- This script prepares predictor variables, such as census data and other socioeconomic indicators. It involves data cleaning and preprocessing.

### 02-Dimensionality Reduction-Predictors Processing.py
-  This script performs dimensionality reduction on the variables that will be used for the "gentrification score". Next, it converts some of the variables to percentages and combines them into the final dataframe that will be used for modelling.
### 03-Transform-Scale.py
-  This script transforms and scales the data to make it suitable for machine learning models.

### 04-Predictions.py
-  This script applies machine learning models to make predictions about gentrification in London's neighborhoods. It includes model training, validation, and predictions. We train a Random Forest, Gradient Boosted Machines and Geographically weighted Random Forest model.

### Visualise data.py
-  This script generates maps to help interpret and understand the data. 

## Dependencies:
- Python 3.10

## Notes:
- Make sure to replace the input data paths with the dataset paths within the scripts.


