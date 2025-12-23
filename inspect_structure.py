import pandas as pd
import ast
import numpy as np
import re
df = pd.read_csv("cleaned_car_data.csv")
print("Original Shape:", df.shape)

# Cleaning Functions
def extract_number(text):
    if pd.isna(text):
        return np.nan
    text = str(text).replace(",", "")
    nums = re.findall(r"[\d\.]+", text)
    return float(nums[0]) if nums else np.nan

def clean_price(price):
    if not price: 
        return np.nan
    price = str(price).lower().replace("₹", "").replace(",", "").strip()
    
    if "lakh" in price:
        num = re.findall(r"[\d\.]+", price)
        return float(num[0]) * 100000 if num else np.nan
    
    nums = re.findall(r"[\d\.]+", price)
    return float(nums[0]) if nums else np.nan

# Convert String Dict to Python Dict
def convert(x):
    try:
        return ast.literal_eval(x)
    except:
        return {}

df["new_car_detail"] = df["new_car_detail"].apply(convert)
df["new_car_overview"] = df["new_car_overview"].apply(convert)
df["new_car_specs"] = df["new_car_specs"].apply(convert)

# Extract from new_car_detail
df["Brand"] = df["new_car_detail"].apply(lambda x: x.get("oem"))
df["Model"] = df["new_car_detail"].apply(lambda x: x.get("model"))
df["Body_Type"] = df["new_car_detail"].apply(lambda x: x.get("bt"))
df["Fuel"] = df["new_car_detail"].apply(lambda x: x.get("ft"))
df["Transmission"] = df["new_car_detail"].apply(lambda x: x.get("transmission"))
df["Owner"] = df["new_car_detail"].apply(lambda x: x.get("owner"))
df["Owner_No"] = df["new_car_detail"].apply(lambda x: x.get("ownerNo"))
df["Year"] = df["new_car_detail"].apply(lambda x: x.get("modelYear"))
df["KM"] = df["new_car_detail"].apply(lambda x: extract_number(x.get("km")))
df["Price"] = df["new_car_detail"].apply(lambda x: clean_price(x.get("price")))

# Extract from Overview (Seats, Engine etc.)
def extract_from_overview(row, key_name):
    try:
        for item in row["top"]:
            if item["key"] == key_name:
                return item["value"]
    except:
        return np.nan
    return np.nan

df["Seats"] = df["new_car_overview"].apply(lambda x: extract_from_overview(x, "Seats"))
df["Seats"] = df["Seats"].apply(lambda x: extract_number(x) if isinstance(x, str) else x)

df["Engine_CC"] = df["new_car_overview"].apply(lambda x: extract_from_overview(x, "Engine Displacement"))
df["Engine_CC"] = df["Engine_CC"].apply(lambda x: extract_number(x) if isinstance(x, str) else x)

# Extract from Specs (Power if exists)
def get_top_spec(x, key):
    try:
        for item in x["top"]:
            if item["key"] == key:
                return item["value"]
    except:
        return np.nan
    return np.nan

df["Power"] = df["new_car_specs"].apply(lambda x: get_top_spec(x, "Max Power"))

# Final Useful Columns
final_df = df[[
    "Brand","Model","Body_Type","Fuel","Transmission","Owner","Owner_No",
    "Year","KM","Price","Seats","Engine_CC","Power","City"
]]

print("Structured Dataset Shape:", final_df.shape)

final_df.to_csv("structured_car_data.csv", index=False)
print("Saved as structured_car_data.csv")

#Result:
#PS C:\Users\USER\OneDrive\Desktop\car dekho> python inspect_structure.py
#Original Shape: (8369, 6)
#Structured Dataset Shape: (8369, 14)
#Saved as structured_car_data.csv

