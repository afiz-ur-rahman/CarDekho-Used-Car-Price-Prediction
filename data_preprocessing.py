import pandas as pd
import numpy as np
import re

# FILE PATHS
chennai_file = r"C:\Users\USER\OneDrive\Desktop\car dekho\chennai_cars.xlsx"
delhi_file = r"C:\Users\USER\OneDrive\Desktop\car dekho\delhi_cars.xlsx"
kolkata_file = r"C:\Users\USER\OneDrive\Desktop\car dekho\kolkata_cars.xlsx"
bangalore_file = r"C:\Users\USER\OneDrive\Desktop\car dekho\bangalore_cars.xlsx"
hyderabad_file = r"C:\Users\USER\OneDrive\Desktop\car dekho\hyderabad_cars.xlsx"
jaipur_file = r"C:\Users\USER\OneDrive\Desktop\car dekho\jaipur_cars.xlsx"
output_file = "cleaned_car_data.csv"

# CLEANING FUNCTIONS
def clean_price(price):
    if pd.isna(price):
        return np.nan
    price = str(price).lower().replace("₹", "").replace(",", "").strip()
    
    if "lakh" in price:
        num = re.findall(r"[\d\.]+", price)
        return float(num[0]) * 100000 if num else np.nan
    num = re.findall(r"[\d\.]+", price)
    return float(num[0]) if num else np.nan

def clean_km(km):
    if pd.isna(km):
        return np.nan
    km = str(km).lower().replace(",", "").replace("kms", "").replace("km", "").strip()
    num = re.findall(r"[\d\.]+", km)
    return float(num[0]) if num else np.nan


def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] >= lower) & (df[column] <= upper)]

# LOAD + TAG CITY
print("Reading files...")

chennai = pd.read_excel(chennai_file)
chennai["City"] = "Chennai"

delhi = pd.read_excel(delhi_file)
delhi["City"] = "Delhi"

kolkata = pd.read_excel(kolkata_file)
kolkata["City"] = "Kolkata"

bangalore = pd.read_excel(bangalore_file)
bangalore["City"] = "Bangalore"

hyderabad = pd.read_excel(hyderabad_file)
hyderabad["City"] = "Hyderabad"

jaipur = pd.read_excel(jaipur_file)
jaipur["City"] = "Jaipur"

# MERGE ALL
df = pd.concat(
    [chennai, delhi, kolkata, bangalore, hyderabad, jaipur],
    ignore_index=True
)
print("Combined Shape:", df.shape)

# RENAME IMPORTANT COLUMNS

rename_map = {
    "year": "Year",
    "Price": "Price",
    "km_driven": "Kilometers",
    "fuel": "Fuel",
    "transmission": "Transmission",
    "owner": "Owner"
}
df.rename(columns=rename_map, inplace=True)

# CLEAN DATA

print("Cleaning Data...")

if "Price" in df.columns:
    df["Price"] = df["Price"].apply(clean_price)

if "Kilometers" in df.columns:
    df["Kilometers"] = df["Kilometers"].apply(clean_km)

if "Year" in df.columns:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

df.drop_duplicates(inplace=True)

# HANDLE MISSING VALUES
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# REMOVE OUTLIERS
if "Price" in df.columns:
    df = df[df["Price"] > 0]
    df = remove_outliers(df, "Price")

if "Kilometers" in df.columns:
    df = df[df["Kilometers"] > 0]


df.reset_index(drop=True, inplace=True)

print("Final Cleaned Shape:", df.shape)

df.to_csv(output_file, index=False)
print("Saved as:", output_file)

#Result:
#PS C:\Users\USER\OneDrive\Desktop\car dekho> python data_preprocessing.py
#Reading files...
#Combined Shape: (8369, 6)
#Cleaning Data...
#Final Cleaned Shape: (8369, 6)
#Saved as: cleaned_car_data.csv
 

