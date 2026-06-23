import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor

# LOAD STRUCTURED DATA
df = pd.read_csv("structured_car_data.csv")
print("Dataset Loaded:", df.shape)

# FEATURES & TARGET
features = [
    "Brand", "Model", "Body_Type", "Fuel", "Transmission",
    "Owner_No", "Year", "KM", "Seats", "Engine_CC", "City"
]
target = "Price"
X = df[features]
y = df[target]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COLUMN TYPES
categorical_cols = ["Brand", "Model", "Body_Type", "Fuel", "Transmission", "City"]
numeric_cols = ["Owner_No", "Year", "KM", "Seats", "Engine_CC"]

# PREPROCESSING PIPELINE
#categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("encoder", BinaryEncoder())])
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OneHotEncoder(handle_unknown="ignore"))])
#categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))])
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
preprocessor = ColumnTransformer(transformers=[("cat", categorical_transformer, categorical_cols),("num", numeric_transformer, numeric_cols)])

# MODEL
#model = RandomForestRegressor(  n_estimators=300,random_state=42,n_jobs=-1)
#model = LinearRegression()
#model = DecisionTreeRegressor(random_state=42)
#model = GradientBoostingRegressor(random_state=42)
model = ExtraTreesRegressor(n_estimators=300,random_state=42,n_jobs=-1)

# FULL PIPELINE
final_model = Pipeline(steps=[("preprocess", preprocessor),("model", model)])

# TRAIN MODEL
final_model.fit(X_train, y_train)
print("Model training completed.")


# MODEL EVALUATION
y_pred = final_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("\n MODEL EVALUATION ")
print("R² Score :", round(r2, 4))
print("MAE      :", round(mae, 2))

# FEATURES USED
print("\nFeatures used:")
print(X.columns.tolist())

# RETRAIN ON FULL DATASET
final_model.fit(X, y)

# SAVE MODEL
joblib.dump(final_model, "car_price_model.pkl")
print("\nModel saved as car_price_model.pkl")

# OneHotEncoder
#Dataset Loaded: (8369, 15)
#Model training completed.
#MODEL EVALUATION
#random forest
#R² Score : 0.9333
#MAE      : 142613.41
#linear regression
#R² Score : 0.5412
#MAE      : 435648.34
#decision tree
#R² Score : 0.9271
#MAE      : 170105.4
#gradient boosting
#R² Score : 0.9338
#MAE      : 210576.87
#extra trees  
#R² Score : 0.9712
#MAE      : 119781.13  

#binary encoder
#random forest
#R² Score : 0.9044
#MAE      : 155050.75
#linear regression
#R² Score : 0.4403
#MAE      : 506724.55
#decision tree
#R² Score : 0.9202
#MAE      : 172680.5
#gradient boosting
#R² Score : 0.8655
#MAE      : 236574.14
#extra trees
#R² Score : 0.9597
#MAE      : 129280.62
#ordinal encoder
#extra trees
#R² Score : 0.9453
#MAE      : 154016.55
#gradient boosting
#R² Score : 0.8808
#MAE      : 229060.28
#decision tree
#R² Score : 0.882
#MAE      : 201774.59
#linear regression
#R² Score : 0.3949
#MAE      : 522124.93
#random forest
#R² Score : 0.893
#MAE      : 165399.34
#target encoding
#pip install category_encoders
#Features used:
#['Brand', 'Model', 'Body_Type', 'Fuel', 'Transmission', 'Owner_No', 'Year', 'KM', 'Seats', 'Engine_CC', 'City']
#Model saved as car_price_model.pkl








