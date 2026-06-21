import pandas as pd
import numpy as np
import ast
import re

# ================================
# LOAD CLEANED DATA
# ================================
df = pd.read_csv("cleaned_car_data.csv")
print("Original Shape:", df.shape)

# ================================
# STRING → DICTIONARY CONVERSION
# ================================
def convert_to_dict(x):

    if isinstance(x, dict):
        return x

    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return {}

    return {}

df["new_car_detail"] = df["new_car_detail"].apply(convert_to_dict)
df["new_car_overview"] = df["new_car_overview"].apply(convert_to_dict)
df["new_car_specs"] = df["new_car_specs"].apply(convert_to_dict)

# ================================
# GENERIC NUMBER EXTRACTOR (KM, SEATS, CC)
# ================================
def get_num(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace(",", "")
    nums = re.findall(r"\d+\.?\d*", x)
    return float(nums[0]) if nums else np.nan

# ================================
# PRICE CLEANING (CRITICAL FIX)
# ================================
def clean_price(x):
    if pd.isna(x):
        return np.nan

    x = str(x).lower().replace(",", "").strip()
    num = float(re.findall(r"\d+\.?\d*", x)[0])

    if "lakh" in x:
        return num * 100000
    elif "crore" in x:
        return num * 10000000
    else:
        return num

# ================================
# PROCESS new_car_detail
# ================================
detail_df = df["new_car_detail"].apply(pd.Series)

df["Brand"] = detail_df["oem"]
df["Model"] = detail_df["model"]
df["Body_Type"] = detail_df["bt"]
df["Fuel"] = detail_df["ft"]
df["Transmission"] = detail_df["transmission"]
df["Owner"] = detail_df["owner"]
df["Owner_No"] = detail_df["ownerNo"]
df["Year"] = detail_df["modelYear"]

df["KM"] = detail_df["km"].apply(get_num)
df["Price"] = detail_df["price"].apply(clean_price)

# ================================
# PROCESS new_car_specs

def get_from_top(data, key):
    if isinstance(data, dict) and "top" in data:
        for item in data["top"]:
            if item.get("key") == key:
                return item.get("value")
    return np.nan

df["Seats"] = df["new_car_specs"].apply(
    lambda x: get_num(get_from_top(x, "Seats"))
)

df["Engine_CC"] = df["new_car_specs"].apply(
    lambda x: get_num(get_from_top(x, "Engine"))
)

df["Mileage"] = df["new_car_specs"].apply(
    lambda x: get_num(get_from_top(x, "Mileage"))
)

df["Power"] = df["new_car_specs"].apply(
    lambda x: get_from_top(x, "Max Power")
)

# ================================
# FINAL STRUCTURED DATASET
# ================================
final_df = df[
    [
        "Brand", "Model", "Body_Type", "Fuel", "Transmission",
        "Owner", "Owner_No", "Year", "KM", "Price",
        "Seats", "Engine_CC", "Power", "Mileage", "City"
    ]
]

print("Final Structured Shape:", final_df.shape)

#  CORRECT FILE NAME
final_df.to_csv("structured_car_data.csv", index=False)
print("Saved as structured_car_data.csv")

print("\nFinal Columns:")
print(final_df.columns)
# Result:
#Original Shape: (8369, 6)
#Final Structured Shape: (8369, 15)
#Saved as structured_car_data.csv
#Final Columns:
#Index(['Brand', 'Model', 'Body_Type', 'Fuel', 'Transmission', 'Owner',
#       'Owner_No', 'Year', 'KM', 'Price', 'Seats', 'Engine_CC', 'Power',
#       'Mileage', 'City'],
#      dtype='object')
