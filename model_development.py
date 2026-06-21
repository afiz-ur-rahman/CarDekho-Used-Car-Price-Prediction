import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

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
final_model.fit(X, y)
print("Model training completed.")

print(final_model.feature_names_in_)

# ================================
# SAVE MODEL
# ================================
joblib.dump(final_model, "car_price_model.pkl")
print("Model saved as car_price_model.pkl")

#Dataset Loaded: (8369, 15)
#Model training completed.
#Model saved as car_price_model.pkl



