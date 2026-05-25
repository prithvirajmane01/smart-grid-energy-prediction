# Smart Grid Energy Demand Prediction using XGBoost

An end-to-end machine learning script that forecasts actual electricity load using historical weather attributes and temporal features. By integrating time-of-day cycles alongside weather inputs, the model achieves high accuracy in mapping out grid demand curves.

## 📊 Project Overview
Efficient energy distribution is a core challenge for modern power grids. This project utilizes an **XGBoost Regressor** workflow to analyze over 35,000 hourly smart grid records, managing overlapping timestamps, missing data, and structural data imbalances to predict target energy consumption.

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.11
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Visualization:** Matplotlib

## 💡 Feature Engineering Insights
Initially, the model relied exclusively on raw weather conditions (`temp`, `humidity`, `wind_speed`, `clouds_all`). To capture human behavioral patterns, I extracted custom temporal structural cycles from the raw timestamps:
* **`hour`**: Captures daily peak usage hours (e.g., evening spikes).
* **`day_of_week`**: Distinguishes commercial/industrial weekday demand from lower weekend cycles.
* **`month`**: Captures broader seasonal changes in climate.

**Impact:** Adding these features drastically improved model performance, reducing the average error down to just **~6.5%** of the target mean load!

## 📈 Visualizations

### 1. Feature Importance
This chart reveals exactly what factors drove the gradient boosting trees. As engineered, the custom time-based parameters play a heavy role alongside climate factors:
![Feature Importance](simple_feature_importance.png)

### 2. Actual vs. Predicted Target Performance
A 100-hour lookback window demonstrating how closely the trained XGBoost model (dashed line) tracks the real-world smart grid energy load (solid line):
![Actual vs Predicted](simple_actual_vs_predicted.png)

## 📉 Results & Performance
* **Dataset Size:** 35,064 entries
* **Target Load Average:** ~28,697 MW
* **Evaluation Metric:** Mean Absolute Error (MAE) = **1,877 MW**
* **Relative Accuracy:** ~93.5% accuracy baseline achieved.

## 🚀 How to Run Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/smart-grid-energy-prediction.git](https://github.com/YOUR_USERNAME/smart-grid-energy-prediction.git)
