# Household-Income-and-Housing-Value-Prediction-

# Overview 

A machine learning project that predicts median house value and household income categories using a decision tree and linear regression model. 

# Methods 

1) Feature engineering
2) Train/Test split
3) Decision tree
4) Linear Regression
5) Evaluated both models using RMSE and accuracy 


# Key Results 

1) For predicting household income categories, a decision tree model was implemented. To increase accuracy from ~60% to ~70%, I created a derived row attribute that recorded the average total number of rooms per community. This increased accuracy by 10%.
2) For predicting median house value, a linear regression model was implemented. I created two geographic-based aggregated derived attributes that group all households into 10 different longitude and latitude ranges, which decreased the RMSE from approximately 68.5k-69.5k to 67.3k-68.0k.

# Technologies 

1) R
2) rpart
3) modelmetrics
4) rpart.plot

# Files 

1) **HousingPrediction.R:** Contains code for a decision tree, linear regression model, derived attributes, and plots.
2) **housing.csv:** CSV file containing the California housing dataset. 


# Dataset Source 

Kaggle Link: https://www.kaggle.com/datasets/camnugent/california-housing-prices






