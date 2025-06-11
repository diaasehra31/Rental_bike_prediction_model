import pandas as pd

# Load the datasets
hour_df = pd.read_csv("hour.csv")
day_df = pd.read_csv("day.csv")

# Display basic info
print("Hourly Data:")
print(hour_df.info())

print("\nDaily Data:")
print(day_df.info())

# Show first few rows
print("\nFirst 5 rows of Hourly Data:")
print(hour_df.head())

print("\nFirst 5 rows of Daily Data:")
print(day_df.head())

# Check for missing values
print("\nMissing Values in Hourly Data:")
print(hour_df.isnull().sum())

print("\nMissing Values in Daily Data:")
print(day_df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows in Hourly Data:", hour_df.duplicated().sum())
print("Duplicate Rows in Daily Data:", day_df.duplicated().sum())

# Check data types
print("\nData Types:")
print(hour_df.dtypes)

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Distribution of bike rentals
plt.figure(figsize=(10,5))
sns.histplot(hour_df['cnt'], bins=50, kde=True, color="blue")
plt.title("Distribution of Total Bike Rentals (Hourly Data)")
plt.xlabel("Total Rentals")
plt.ylabel("Count")
plt.show()

# Bike rentals by season
plt.figure(figsize=(8,5))
sns.boxplot(x="season", y="cnt", hue="season", data=hour_df, palette="coolwarm", legend=False)
plt.title("Bike Rentals by Season")
plt.xlabel("Season (1:Spring, 2:Summer, 3:Fall, 4:Winter)")
plt.ylabel("Total Rentals")
plt.show()

# Bike rentals by hour
plt.figure(figsize=(12,6))
sns.lineplot(x="hr", y="cnt", data=hour_df, errorbar=None, marker="o", color="red")
plt.title("Bike Rentals by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Total Rentals")
plt.xticks(range(0,24))
plt.show()

# Convert categorical features to category type
hour_df['season'] = hour_df['season'].astype('category')
hour_df['yr'] = hour_df['yr'].astype('category')
hour_df['mnth'] = hour_df['mnth'].astype('category')
hour_df['holiday'] = hour_df['holiday'].astype('category')
hour_df['weekday'] = hour_df['weekday'].astype('category')
hour_df['workingday'] = hour_df['workingday'].astype('category')
hour_df['weathersit'] = hour_df['weathersit'].astype('category')

# Drop irrelevant columns
hour_df = hour_df.drop(columns=['instant', 'dteday'])  # 'instant' is an index, 'dteday' is date (not needed)

# Create new features
hour_df['rush_hour'] = hour_df['hr'].apply(lambda x: 1 if x in [7, 8, 17, 18] else 0)  # Peak hours

print("\nTransformed Dataset:")
print(hour_df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Select features and target variable
X = hour_df.drop(columns=['cnt', 'casual', 'registered'])  # Drop target & redundant cols
y = hour_df['cnt']  # Target variable

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

import numpy as np

# Compute RMSE manually
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

print("R² Score:", r2_score(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
print("\nRandom Forest Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R² Score:", r2_score(y_test, y_pred_rf))

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train best model
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

# Evaluate best model
print("\nTuned Random Forest Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred_best_rf))
print("MSE:", mean_squared_error(y_test, y_pred_best_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_best_rf)))
print("R² Score:", r2_score(y_test, y_pred_best_rf))

import pickle

# Save the trained model
with open("bike_rental_model.pkl", "wb") as file:
    pickle.dump(best_rf, file)

print("\n Model saved as 'bike_rental_model.pkl'")

