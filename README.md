# CarDekho Used Car Price Prediction

## Project Overview

This project aims to predict the selling price of used cars using Machine Learning techniques. The dataset was collected from CarDekho and contains information such as car brand, model, fuel type, transmission type, year of manufacture, kilometers driven, engine capacity, seating capacity, and city.

The project covers the complete Machine Learning workflow including data preprocessing, feature engineering, model development, model evaluation, and deployment using Streamlit.

---

## Problem Statement

Determining the correct price of a used car is a challenging task due to multiple influencing factors such as brand, age, mileage, engine specifications, and ownership history.

The objective of this project is to build a machine learning model capable of accurately predicting used car prices based on vehicle attributes.

---

## Dataset Information

### Dataset Size

* Total Records: 8,369
* Features Used: 11
* Target Variable: Price

### Features

| Feature      | Description               |
| ------------ | ------------------------- |
| Brand        | Car manufacturer          |
| Model        | Car model                 |
| Body_Type    | Vehicle body type         |
| Fuel         | Fuel type                 |
| Transmission | Transmission type         |
| Owner_No     | Number of previous owners |
| Year         | Manufacturing year        |
| KM           | Kilometers driven         |
| Seats        | Seating capacity          |
| Engine_CC    | Engine displacement       |
| City         | City where car is listed  |

### Target Variable

* Price

---

## Data Preprocessing

The following preprocessing steps were performed:

* Combined multiple city datasets
* Converted nested JSON-like columns into dictionaries
* Extracted useful attributes from nested data
* Cleaned price values
* Extracted numerical values from KM, Seats, and Engine specifications
* Handled missing values using SimpleImputer
* Encoded categorical features using OneHotEncoder

---

## Exploratory Data Analysis

EDA techniques used:

* Dataset inspection
* Missing value analysis
* Feature distribution analysis
* Correlation analysis
* Outlier identification
* Feature relationship analysis

---

## Machine Learning Models Evaluated

### Linear Regression

* R² Score: 0.5412
* MAE: ₹435,648

### Decision Tree Regressor

* R² Score: 0.9271
* MAE: ₹170,105

### Random Forest Regressor

* R² Score: 0.9333
* MAE: ₹142,613

### Gradient Boosting Regressor

* R² Score: 0.9338
* MAE: ₹210,577

### Extra Trees Regressor (Final Model)

* R² Score: 0.9712
* MAE: ₹119,781

---

## Model Selection

Multiple regression algorithms were evaluated and compared using R² Score and Mean Absolute Error (MAE).

The Extra Trees Regressor achieved the highest predictive performance and was selected as the final model.

### Final Model

* Algorithm: Extra Trees Regressor
* Encoder: OneHotEncoder
* R² Score: 0.9712
* MAE: ₹119,781

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit

---

## Project Structure

```text
Car-Dekho-Price-Prediction/
│
├── data_preprocessing.py
├── model_development.py
├── app.py
├── structured_car_data.csv
├── car_price_model.pkl
├── requirements.txt
├── README.md
│
└── notebooks/
```

## Model Deployment

The trained machine learning model was deployed using Streamlit.

Users can:

* Select car attributes
* Enter vehicle specifications
* Predict estimated used car price instantly

---

## Skills Demonstrated

* Data Cleaning
* Data Preprocessing
* Feature Engineering
* Exploratory Data Analysis
* Categorical Encoding
* Machine Learning Model Development
* Model Evaluation
* Model Comparison
* Hyperparameter Optimization
* Model Deployment
* Streamlit Application Development

---

## Results

The final Extra Trees Regressor model demonstrated excellent predictive performance:

* R² Score: 0.9712
* MAE: ₹119,781

This indicates that the model can accurately estimate used car prices and generalizes well to unseen data.

---

## Conclusion

A complete end-to-end machine learning solution was developed for predicting used car prices using CarDekho data. After evaluating multiple regression algorithms, Extra Trees Regressor combined with OneHotEncoder delivered the best performance. The model was successfully deployed through Streamlit, enabling users to obtain real-time price predictions through an interactive web application.



