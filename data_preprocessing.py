import pandas as pd
import numpy as np
import re

# FILE PATHS
chennai_file   = r"C:\Users\USER\OneDrive\Desktop\car dekho\chennai_cars.xlsx"
delhi_file     = r"C:\Users\USER\OneDrive\Desktop\car dekho\delhi_cars.xlsx"
kolkata_file   = r"C:\Users\USER\OneDrive\Desktop\car dekho\kolkata_cars.xlsx"
bangalore_file = r"C:\Users\USER\OneDrive\Desktop\car dekho\bangalore_cars.xlsx"
hyderabad_file = r"C:\Users\USER\OneDrive\Desktop\car dekho\hyderabad_cars.xlsx"
jaipur_file    = r"C:\Users\USER\OneDrive\Desktop\car dekho\jaipur_cars.xlsx"
output_file = "cleaned_car_data.csv"

# CLEANING FUNCTIONS
def clean_price(price):
    if pd.isna(price):
        return np.nan
    price = str(price).lower().replace("₹", "").replace(",", "").strip()
    num_parts = re.findall(r"[\d\.]+", price)
    if not num_parts:
        return np.nan
    num_str = "".join(num_parts)
    if "lakh" in price:
        return float(num_str) * 100000
    return float(num_str)

def clean_km(km):
    if pd.isna(km):
        return np.nan
    km = str(km).lower().replace(",", "").replace("kms", "").replace("km", "").strip()
    num_parts = re.findall(r"[\d\.]+", km)
    if not num_parts:
        return np.nan
    return float("".join(num_parts))

# READ FILES + TAG CITY
print("Reading files...")

chennai   = pd.read_excel(chennai_file)
chennai["City"] = "Chennai"

delhi     = pd.read_excel(delhi_file)
delhi["City"] = "Delhi"

kolkata   = pd.read_excel(kolkata_file)
kolkata["City"] = "Kolkata"

bangalore = pd.read_excel(bangalore_file)
bangalore["City"] = "Bangalore"

hyderabad = pd.read_excel(hyderabad_file)
hyderabad["City"] = "Hyderabad"

jaipur    = pd.read_excel(jaipur_file)
jaipur["City"] = "Jaipur"

# MERGE ALL
df = pd.concat([chennai, delhi, kolkata, bangalore, hyderabad, jaipur], ignore_index=True)
print("Combined Shape:", df.shape)

print("\nColumn names:") 
print(df.columns) 
print(df.head())
print("\nDataset info:") 
print(df.info())

# SAVE TO CSV
df.to_csv(output_file, index=False)
print("Saved cleaned file as:", output_file)

#Result:
#PS C:\Users\USER\OneDrive\Desktop\car dekho> python data_preprocessing.py
#Reading files...
#Combined Shape: (8369, 6)

#Column names:
#Index(['new_car_detail', 'new_car_overview', 'new_car_feature',
#       'new_car_specs', 'car_links', 'City'],
#      dtype='object')
 #                                     new_car_detail  ...     City
0 # {'it': 0, 'ft': 'Petrol', 'bt': 'SUV', 'km': '...  ...  Chennai
1 # {'it': 0, 'ft': 'Petrol', 'bt': 'Minivans', 'k...  ...  Chennai
2 # {'it': 0, 'ft': 'Petrol', 'bt': 'SUV', 'km': '...  ...  Chennai
3 # {'it': 0, 'ft': 'Petrol', 'bt': 'Hatchback', '...  ...  Chennai
4 # {'it': 0, 'ft': 'Petrol', 'bt': 'Hatchback', '...  ...  Chennai

#[5 rows x 6 columns]

#Dataset info:
#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 8369 entries, 0 to 8368
#Data columns (total 6 columns):
 #   Column            Non-Null Count  Dtype
#---  ------            --------------  -----
 #0   new_car_detail    8369 non-null   object
 #1   new_car_overview  8369 non-null   object
 #2   new_car_feature   8369 non-null   object
 #3   new_car_specs     8369 non-null   object
 #4   car_links         8369 non-null   object
 #5   City              8369 non-null   object
#dtypes: object(6)
#memory usage: 392.4+ KB
#None
#Saved cleaned file as: cleaned_car_data.csv
 


