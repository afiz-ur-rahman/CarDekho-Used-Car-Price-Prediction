Car Dekho – Used Car Price Prediction
 Project Overview

This project predicts the resale price of used cars using Machine Learning based on real CarDekho dataset.
A Streamlit web application is developed to allow users to input car details and instantly receive an estimated price.

Objective

Build an accurate ML model
 Evaluate models and select best performer
 Deploy price prediction as a web application

 Dataset

Source: CarDekho
Records: 8369
Cities: Chennai, Delhi, Kolkata, Bangalore, Hyderabad, Jaipur

Features Used

Brand

Model

Body Type

Fuel

Transmission

Owner Number

Year

Kilometers Driven

Seats

Engine CC

City

Data Processing

Combined multi-city dataset
Converted JSON format to tabular data
 Handled missing values
 Removed outliers
 Structured ML-ready dataset

 EDA

 Price distribution
 Year vs Price
 Fuel & Transmission impact
 Correlation heatmap

 Model Development

Models trained:

Linear Regression

Random Forest

Gradient Boosting

 Best Model

Random Forest Regressor

 Final Accuracy

R² Score: 0.89

MAE: ₹1.41 Lakhs

RMSE: ₹3.84 Lakhs
