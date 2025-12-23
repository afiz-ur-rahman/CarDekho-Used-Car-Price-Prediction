import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib

# LOAD DATA
df = pd.read_csv("structured_car_data.csv")
print("Dataset Loaded:", df.shape)
# FEATURES & TARGET
target = "Price"
features = [
    "Brand","Model","Body_Type","Fuel","Transmission",
    "Owner_No","Year","KM","Seats","Engine_CC","City"
]
X = df[features]
y = df[target]

# CATEGORICAL & NUMERIC COLUMNS
categorical_cols = [
    "Brand","Model","Body_Type","Fuel","Transmission","City"
]
numeric_cols = [
    "Owner_No","Year","KM","Seats","Engine_CC"
]

# PREPROCESSING WITH IMPUTATION
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols)
    ]
)

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# MODELS
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}
results = {}

# TRAIN + EVALUATE
for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }
    print("\n==============================")
    print(name)
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)
    
# SELECT BEST MODEL
best_model = max(results, key=lambda x: results[x]["R2 Score"])
print("\nBest Model Selected =", best_model)
final_model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", models[best_model])
    ]
)
final_model.fit(X, y)
# SAVE MODEL
joblib.dump(final_model, "car_price_model.pkl")
print("\nModel Saved as car_price_model.pkl")
#Result:
#PS C:\Users\USER\OneDrive\Desktop\car dekho> python model_development.py
#Dataset Loaded: (8369, 14)
#Train Shape: (6695, 11)
#Test Shape: (1674, 11)
#Linear Regression
#MAE : 370634.0805576693
#MSE : 513893245947.7109
#RMSE: 716863.4779005769
#R2  : 0.6304124696095292
#Random Forest
#MAE : 141843.38217283352
#MSE : 147724203294.70917
#RMSE: 384349.0643864106
#R2  : 0.89375804429205
#Gradient Boosting
#MAE : 215760.34839864986
#MSE : 197913166647.3155
#RMSE: 444874.3268017559
#R2  : 0.8576625805656498
#Best Model Selected = Random Forest



