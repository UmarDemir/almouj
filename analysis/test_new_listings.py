"""Test the saved MLR and PCR models on 10 new OpenSooq listings.

Run after model.py:  python analysis/test_new_listings.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
MODELS = BASE / 'models'
df_new = pd.read_csv(BASE / 'data' / 'new_listings_opensooq.csv')

model_features = joblib.load(MODELS / 'feature_list.pkl')
dropped_columns = joblib.load(MODELS / 'Dropping_Dummies.pkl')

df_new_dummies = pd.get_dummies(df_new, columns=['Location', 'View', 'Furnishing Status'], 
                                drop_first=False, dtype=float)

for feature in model_features:
    if feature not in df_new_dummies.columns:
        df_new_dummies[feature] = 0

columns_to_keep = [col for col in model_features if col not in dropped_columns]
X_new = df_new_dummies[columns_to_keep] 

print("\nNew data features after alignment:")
print(X_new.columns.tolist())

mlr_model = joblib.load(MODELS / 'mlr_model.pkl')
X_new_with_const = sm.add_constant(X_new, has_constant='add')
log_predictions_mlr = mlr_model.predict(X_new_with_const)
predictions_mlr = np.exp(log_predictions_mlr)

df_new['Predicted_Price_MLR'] = predictions_mlr
df_new['MLR_Error'] = df_new['Price (OMR)'] - df_new['Predicted_Price_MLR']

scaler = joblib.load(MODELS / 'standard_scaler.pkl')
pca = joblib.load(MODELS / 'pca_model.pkl')
pcr_model = joblib.load(MODELS / 'pcr_model.pkl')

X_new_scaled = scaler.transform(X_new)
X_new_pca = pca.transform(X_new_scaled)
X_new_pca_const = sm.add_constant(X_new_pca, has_constant='add')
log_predictions_pcr = pcr_model.predict(X_new_pca_const)
predictions_pcr = np.exp(log_predictions_pcr)

df_new['Predicted_Price_PCR'] = predictions_pcr
df_new['PCR_Error'] = df_new['Price (OMR)'] - df_new['Predicted_Price_PCR']

pd.set_option('display.float_format', '{:,.2f}'.format)
print("\nPredictions on New Data:")
print(df_new[['Price (OMR)', 'Predicted_Price_MLR', 'Predicted_Price_PCR', 'MLR_Error', 'PCR_Error']])

from sklearn.metrics import mean_absolute_error, mean_squared_error

mlr_mae = mean_absolute_error(df_new['Price (OMR)'], df_new['Predicted_Price_MLR'])
pcr_mae = mean_absolute_error(df_new['Price (OMR)'], df_new['Predicted_Price_PCR'])

mlr_rmse = np.sqrt(mean_squared_error(df_new['Price (OMR)'], df_new['Predicted_Price_MLR']))
pcr_rmse = np.sqrt(mean_squared_error(df_new['Price (OMR)'], df_new['Predicted_Price_PCR']))

print(f"\nModel Performance on NEW DATA:")
print(f"{'Metric':<20} {'MLR':<10} {'PCR':<10}")
print(f"{'MAE (OMR)':<20} {mlr_mae:<10.2f} {pcr_mae:<10.2f}")
print(f"{'RMSE (OMR)':<20} {mlr_rmse:<10.2f} {pcr_rmse:<10.2f}")



