import pandas as pd

print("loading data")
data_energy=pd.read_csv("energy_dataset.csv")
data_weather=pd.read_csv("weather_features.csv")

print("preparing data")
data_weather=data_weather.rename(columns={"dt_iso":"time"})

data_energy["time"]=pd.to_datetime(data_energy["time"],utc=True)
data_weather["time"]=pd.to_datetime(data_weather["time"],utc=True)

weather_features_to_keep=['time', 'temp', 'humidity', 'wind_speed', 'clouds_all']
data_weather=data_weather[weather_features_to_keep]



data_weather_hourly=data_weather.groupby('time').mean().reset_index()

final_df = pd.merge(data_energy, data_weather_hourly, on='time', how='inner')

features_to_keep = ['time','total load actual','temp','humidity','wind_speed','clouds_all']
final_df = final_df[features_to_keep]
final_df = final_df.ffill()

final_df['hour'] = final_df['time'].dt.hour
final_df['day_of_week'] = final_df['time'].dt.dayofweek 
final_df['month'] = final_df['time'].dt.month

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


print(final_df.columns)
X=final_df[["temp", "humidity", "wind_speed", "clouds_all", "hour", "day_of_week", "month"]]
y=final_df["total load actual"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=XGBRegressor(n_estimators=100,learning_rate=0.1,max_depth=4)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(y_pred)

mae=mean_absolute_error(y_test,y_pred)
# print(final_df["total load actual"].describe())
# print(y.describe())
print(mae)

import matplotlib.pyplot as plt
import numpy as np

def create_visualizations(X, model, y_test, y_pred):
    import matplotlib.pyplot as plt
    import numpy as np

    # ==========================================
    # 1. SIMPLE FEATURE IMPORTANCE PLOT
    # ==========================================
    print("Generating Feature Importance Plot...")
    
    # Get and sort importances
    importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(importances) # Sorted lowest to highest for a nice bottom-up bar chart

    plt.figure(figsize=(10, 5))
    plt.title("Feature Importance (XGBoost)")
    
    # Simple horizontal bar chart
    plt.barh(features[indices], importances[indices], color="royalblue")
    plt.xlabel("Importance Score")
    
    plt.tight_layout()
    plt.savefig("simple_feature_importance.png")
    plt.close()


    # ==========================================
    # 2. SIMPLE ACTUAL VS PREDICTED PLOT
    # ==========================================
    print("Generating Actual vs Predicted Plot...")
    
    # Convert y_test to a simple numpy array so indexing is straightforward
    y_test_array = np.array(y_test)

    plt.figure(figsize=(12, 5))
    plt.title("Actual vs Predicted Energy Load (100 Hour Snapshot)")
    
    # Simply plot the first 100 points as lines
    plt.plot(y_test_array[:100], label="Actual Data", color="tab:blue", linewidth=2)
    plt.plot(y_pred[:100], label="Predicted Data", color="tab:orange", linestyle="--", linewidth=2)
    
    plt.xlabel("Hours")
    plt.ylabel("Energy Load (MW)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("simple_actual_vs_predicted.png")
    plt.close()

    print("Done! Check your folder for 'simple_feature_importance.png' and 'simple_actual_vs_predicted.png'!")

create_visualizations(X, model, y_test, y_pred)