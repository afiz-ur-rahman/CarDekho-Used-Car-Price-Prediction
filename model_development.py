import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ================================
# LOAD STRUCTURED DATA
# ================================
df = pd.read_csv("structured_car_data.csv")
print("Dataset Loaded:", df.shape)



# ================================
# FEATURES & TARGET
# ================================
features = [
    "Brand", "Model", "Body_Type", "Fuel", "Transmission",
    "Owner_No", "Year", "KM", "Seats", "Engine_CC", "City"
]
target = "Price"

X = df[features]
y = df[target]

# ================================
# SPLIT DATA
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# COLUMN TYPES
# ================================
categorical_cols = [
    "Brand", "Model", "Body_Type", "Fuel", "Transmission", "City"
]

numeric_cols = [
    "Owner_No", "Year", "KM", "Seats", "Engine_CC"
]

# ================================
# PREPROCESSING PIPELINE
# ================================
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

# ================================
# RANDOM FOREST 
# MODEL
# ================================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# ================================
# FULL PIPELINE
# ================================
final_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

# ================================
# TRAIN MODEL
# ================================
final_model.fit(X_train, y_train)
print("Model training completed.")

# ================================
# MODEL EVALUATION
# ================================
y_pred = final_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print("R² Score :", round(r2, 4))
print("MAE      :", round(mae, 2))

# ================================
# FEATURES USED
# ================================
print("\nFeatures used:")
print(X.columns.tolist())

# ================================
# RETRAIN ON FULL DATASET
# ================================
final_model.fit(X, y)

# ================================
# SAVE MODEL
# ================================
joblib.dump(final_model, "car_price_model.pkl")
print("\nModel saved as car_price_model.pkl")

#Dataset Loaded: (8369, 15)
#Model training completed.
#===== MODEL EVALUATION =====
#R² Score : 0.9333
#MAE      : 142613.41
#Features used:
#['Brand', 'Model', 'Body_Type', 'Fuel', 'Transmission', 'Owner_No', 'Year', 'KM', 'Seats', 'Engine_CC', 'City']
#Model saved as car_price_model.pkl









