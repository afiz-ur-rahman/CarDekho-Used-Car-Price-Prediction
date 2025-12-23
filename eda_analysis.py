import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("structured_car_data.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nStatistical Summary:\n")
print(df.describe())

# PRICE DISTRIBUTION
plt.figure(figsize=(8,5))
sns.histplot(df["Price"], kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.savefig("price_distribution.png")
plt.show()

# YEAR DISTRIBUTION

plt.figure(figsize=(8,5))
sns.histplot(df["Year"], bins=20)
plt.title("Car Manufacturing Year Distribution")
plt.savefig("year_distribution.png")
plt.show()


# KM DRIVEN

plt.figure(figsize=(8,5))
sns.histplot(df["KM"], kde=True)
plt.title("Kilometers Driven Distribution")
plt.savefig("km_distribution.png")
plt.show()


# PRICE vs YEAR

plt.figure(figsize=(8,6))
sns.boxplot(x=df["Year"], y=df["Price"])
plt.xticks(rotation=90)
plt.title("Price vs Year")
plt.savefig("price_vs_year.png")
plt.show()


# PRICE vs FUEL

plt.figure(figsize=(7,5))
sns.boxplot(x="Fuel", y="Price", data=df)
plt.title("Price vs Fuel Type")
plt.savefig("price_vs_fuel.png")
plt.show()


# PRICE vs TRANSMISSION

plt.figure(figsize=(7,5))
sns.boxplot(x="Transmission", y="Price", data=df)
plt.title("Price vs Transmission")
plt.savefig("price_vs_transmission.png")
plt.show()


# CORRELATION HEATMAP

plt.figure(figsize=(7,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

print("\nEDA Completed Successfully. Plots Saved.")

#PS C:\Users\USER\OneDrive\Desktop\car dekho> python eda_analysis.py
#Dataset Shape: (8369, 14)

#Columns:
 #Index(['Brand', 'Model', 'Body_Type', 'Fuel', 'Transmission', 'Owner',
       #'Owner_No', 'Year', 'KM', 'Price', 'Seats', 'Engine_CC', 'Power',
       #'City'],
      #dtype='object')

#Missing Values:

#Brand            0
#Model            0
#Body_Type        4
#Fuel             0
#Transmission     0
#Owner            0
#Owner_No         0
#Year             0
#KM               0
#Price            0
#Seats            6
#Engine_CC        4
#Power           60
#City             0
#dtype: int64

#Statistical Summary:

 #         Owner_No         Year            KM         Price        Seats    Engine_CC
#count  8369.000000  8369.000000  8.369000e+03  8.369000e+03  8363.000000  8365.000000
#mean      1.360139  2016.503286  5.897430e+04  9.142701e+05     5.203276  1424.735923
#std       0.641958     3.921465  7.406100e+04  1.061816e+06     0.663789   477.629144
#min       0.000000  1985.000000  0.000000e+00  1.030000e+00     2.000000     0.000000
#25%       1.000000  2014.000000  3.000000e+04  3.980000e+05     5.000000  1197.000000
#50%       1.000000  2017.000000  5.369200e+04  6.000000e+05     5.000000  1248.000000
#75%       2.000000  2019.000000  8.000000e+04  9.500000e+05     5.000000  1498.000000
#max       5.000000  2023.000000  5.500000e+06  9.600000e+06    10.000000  5000.000000

#EDA Completed Successfully. Plots Saved.
