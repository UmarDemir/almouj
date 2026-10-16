"""Almouj hedonic pricing model: data preparation, MLR and PCR fitting.

Originally written for an IB Mathematics AA HL Internal Assessment (2025).
Run from the repository root:  python analysis/model.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.decomposition import PCA


sns.set_style("whitegrid")
BASE = Path(__file__).resolve().parent
MODELS = BASE / 'models'
MODELS.mkdir(exist_ok=True)
df = pd.read_csv(BASE / 'data' / 'almouj_data.csv') 

categorical_columns = ['Location', 'View', 'Furnishing Status']


#2.2
df_with_dummies = pd.get_dummies(df, columns=categorical_columns, drop_first=False, dtype=float)
all_view_columns = [col for col in df_with_dummies.columns if col.startswith('View_')]
view_columns_to_drop = ['View_Other']

location_columns = [col for col in df_with_dummies.columns if 'Location_' in col]
if location_columns:

    location_columns_to_drop = sorted(location_columns)[0]
else:
    location_columns_to_drop = []

furnishing_columns = [col for col in df_with_dummies.columns if 'Furnish' in col]
if furnishing_columns:
    furnishing_columns_to_drop = sorted(furnishing_columns)[0]
else:
    furnishing_columns_to_drop = []

columns_to_drop = view_columns_to_drop + [location_columns_to_drop, furnishing_columns_to_drop]
columns_to_drop = [col for col in columns_to_drop if col]
df_with_dummies = df_with_dummies.drop(columns=columns_to_drop)

'''this section was removed as it made the model Prediction power worse.
# Remove the 'Bedrooms' column to address multicollinearity
df_with_dummies = df_with_dummies.drop('Bedrooms', axis=1)
print("'Bedrooms' feature removed to reduce multicollinearity.")
df_with_dummies = df_with_dummies.drop('Bathrooms', axis=1)
print("'Bathrooms' feature removed to further reduce multicollinearity.")
'''

df_with_dummies = df_with_dummies.drop('Parking Spaces', axis=1)

y = df_with_dummies['Log_Price']
X = df_with_dummies.drop(['Log_Price'], axis=1)


#2.3
corr_matrix = df_with_dummies.corr()

log_price_correlations = corr_matrix['Log_Price'].abs().sort_values(ascending=False)
print(" ")
print("Top 15 features correlated with Log_Price:")
print(log_price_correlations.head(15))
print("\n")

print("\n4. Analysis of Potential Multicollinearity (VIF > 5):")
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_vals = []
for i in range(len(X.columns)):
    vif_vals.append(variance_inflation_factor(X.values, i))
vif_data["VIF"] = vif_vals
high_vif = vif_data[vif_data['VIF'] > 5]
print(high_vif)


#3.1
print ("\n3.1 Multiple Linear Regression (MLR)\n")

X_with_const = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(X_with_const, y, test_size=0.2, random_state=42)
model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MODEL SUMMARY")
print(model.summary())

coefficients_df = pd.DataFrame({
    'Feature': model.params.index,
    'Coefficient': model.params.values,
    'P-value': model.pvalues.values
})
significant_coeffs = coefficients_df[coefficients_df['P-value'] < 0.05].sort_values('Coefficient', ascending=False)

for index, row in significant_coeffs.head(5).iterrows():
    feature = row['Feature']
    coef = row['Coefficient']

    if feature != 'const':
        print(f"- {feature}: {coef:.4f} → ~{(coef * 100):.2f}%")



#3.2
print ("\n3.2 Principal Component Regression (PCR)\n")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA() # Applying PCA
X_pca = pca.fit_transform(X_scaled)

feature_names = X.columns.tolist()
loadings = pca.components_

pca_loadings_df = pd.DataFrame(loadings[:5, :], columns=feature_names, index=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
pca_loadings_df_rounded = pca_loadings_df.round(3)
print("\n The First 5 Principal Components:")
print(pca_loadings_df_rounded)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("PCA Explained Variance Ratio:")
for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i+1}: {ev:.4f} (Cumulative: {cv:.4f})")

n_components = np.argmax(cumulative_variance >= 0.95) + 1
print("\nPCR Model:")
print(f"Selected {n_components} components explain {cumulative_variance[n_components-1]:.2%} variance")

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

X_train_pca_const = sm.add_constant(X_train_pca)
X_test_pca_const = sm.add_constant(X_test_pca)

pcr_model = sm.OLS(y_train_pca, X_train_pca_const).fit()
y_pred_pcr = pcr_model.predict(X_test_pca_const)

pcr_r2 = r2_score(y_test_pca, y_pred_pcr)
pcr_rmse = np.sqrt(mean_squared_error(y_test_pca, y_pred_pcr))

print("\nPCR Model Summary:")
print(pcr_model.summary())


#4.1
print("\n4.1 Testing Phase on New Data\n")

adj_r2_mlr = model.rsquared_adj
adj_r2_pcr = pcr_model.rsquared_adj


r2_mlr_test = r2_score(y_test, y_pred)
r2_pcr_test = r2_score(y_test_pca, y_pred_pcr)

print("MODEL COMPARISON:")
print(f"{'Metric':<30} {'MLR':<10} {'PCR':<10}")
print(f"{'R-squared':<30} {r2_mlr_test:.4f}    {r2_pcr_test:.4f}")
print(f"{'Adjusted R-squared':<30} {adj_r2_mlr:.4f}    {adj_r2_pcr:.4f}")
print(f"{'RMSE to the log scale':<30} {rmse:.4f}    {pcr_rmse:.4f}")
print(f"{'Approximate OMR Error':<30} {rmse * 165410:.2f}    {pcr_rmse * 165410:.2f}")


# those models are saved to be used in 4.2 testing phase
joblib.dump(model, MODELS / 'mlr_model.pkl')
joblib.dump(pcr_model, MODELS / 'pcr_model.pkl')
joblib.dump(scaler, MODELS / 'standard_scaler.pkl')
joblib.dump(pca, MODELS / 'pca_model.pkl')
joblib.dump(X.columns.tolist(), MODELS / 'feature_list.pkl')
joblib.dump(columns_to_drop, MODELS / 'Dropping_Dummies.pkl')