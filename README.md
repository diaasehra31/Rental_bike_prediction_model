# 🚲 Bike Rental Demand Prediction

This project predicts the number of bikes rented in a day based on historical and environmental data. It uses machine learning techniques and is deployed via a Flask web application for real-time predictions.

---

## 📌 Problem Statement

Bike rental businesses often struggle with imbalanced bike distribution due to unpredictable demand. The objective of this project is to **accurately predict daily rental demand** using past records, weather conditions, seasonality, and user behavior.

---

## 📁 Dataset

The dataset is taken from the UCI Machine Learning Repository.

- **day.csv**: Contains daily data (used for modeling)
- **hour.csv**: Contains hourly data (used for EDA and visualization)

**Key Features Used:**

- `season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday`, `weathersit`
- `temp`, `atemp`, `hum`, `windspeed`
- `casual`, `registered`

**Target Variable:**
- `cnt` (total count of rented bikes)

---

## ⚙️ Tech Stack

| Type        | Tools/Libraries                                      |
|-------------|------------------------------------------------------|
| Language    | Python                                               |
| Data Handling | Pandas, NumPy                                     |
| Visualization | Matplotlib, Seaborn                              |
| Machine Learning | scikit-learn                                  |
| Backend     | Flask                                                |
| Frontend    | HTML, CSS, JavaScript                                |
| Deployment  | Localhost (Flask), Pickle                            |

---

## 🔍 Exploratory Data Analysis

- **Distribution** of rentals
- **Seasonal trends** (Spring vs Winter, etc.)
- **Rush hour impact** using new engineered feature
- **Hourly vs Daily rental pattern**
- **Weather and temperature effect**

Visualizations created using `Matplotlib` and `Seaborn`.

---

## 🤖 Model Building

Models trained:

1. **Linear Regression**
2. **Random Forest Regressor** ✅ *(Best Performing)*

### 🧪 Metrics Used:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

### ✅ Final Model (Tuned Random Forest):

| Metric  | Value        |
|---------|--------------|
| MAE     | ~35          |
| RMSE    | ~50          |
| R²      | ~0.95        |

### 📦 Model Export:

The model is saved using `pickle` as `bike_rental_model.pkl`.

---

## 🌐 Web Application

Built using **Flask**, the application allows users to input features and get predicted demand.

### 🔧 How to Run:

```bash
# Step 1: Clone the repository
git clone https://github.com/your-username/bike-rental-prediction.git
cd bike-rental-prediction

# Step 2: Install requirements
pip install -r requirements.txt

# Step 3: Run the Flask App
python app.py
Go to: http://localhost:5000/

