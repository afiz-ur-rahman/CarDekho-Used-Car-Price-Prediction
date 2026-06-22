# Car Dheko - Used Car Price Prediction

## Project Overview

The Car Dheko Used Car Price Prediction project is an end-to-end Machine Learning application developed to estimate the selling price of used cars based on vehicle specifications. The project uses a Random Forest Regression model and provides real-time price predictions through a Streamlit web application.

---

## Problem Statement

Used car prices depend on several factors such as brand, model, fuel type, transmission type, manufacturing year, kilometers driven, engine capacity, ownership history, and city. The objective of this project is to develop a machine learning model that can accurately predict the price of a used car using these features.

---

## Dataset Information

* Dataset Size: 8,369 Records
* Target Variable: Price

### Features Used

* Brand
* Model
* Body_Type
* Fuel
* Transmission
* Owner_No
* Year
* KM
* Seats
* Engine_CC
* City

---

## Project Workflow

### 1. Data Cleaning & Preprocessing

* Combined datasets from multiple cities.
* Handled missing values using SimpleImputer.
* Encoded categorical features using OneHotEncoder.
* Prepared structured data for machine learning.

### 2. Exploratory Data Analysis (EDA)

Performed analysis to understand:

* Brand Distribution
* Fuel Type Distribution
* Vehicle Price Distribution
* Manufacturing Year Trends
* Kilometers Driven Analysis
* Correlation Heatmap

### 3. Feature Selection

Selected important features influencing used car prices:

* Brand
* Model
* Body Type
* Fuel Type
* Transmission
* Ownership History
* Year
* Kilometers Driven
* Seating Capacity
* Engine Capacity
* City

### 4. Machine Learning Model Development

Algorithm Used:

* Random Forest Regressor

Model Configuration:

* n_estimators = 300
* random_state = 42
* n_jobs = -1

### 5. Model Evaluation

Evaluation Metrics:

* R² Score = 0.9333
* MAE (Mean Absolute Error) = ₹142,613.41

The model achieved strong predictive performance for used car price estimation.

### 6. Model Deployment

The trained model was saved using Joblib and deployed using Streamlit.

Deployment Workflow:

User Input → Model Prediction → Estimated Car Price

### 7. Streamlit Application

Features:

* Brand Selection
* Model Selection
* Body Type Selection
* Fuel Type Selection
* Transmission Selection
* Owner Number Input
* Manufacturing Year Input
* Kilometers Driven Input
* Seating Capacity Input
* Engine Capacity Input
* City Selection
* Real-Time Price Prediction

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit
* Matplotlib
* Seaborn

---

## Skills Demonstrated

### Data Cleaning & Preprocessing

Handled missing values and prepared data for machine learning.

### Exploratory Data Analysis (EDA)

Identified patterns, trends, and feature relationships using visualizations.

### Feature Engineering

Selected relevant features affecting used car prices.

### Machine Learning Model Development

Built and trained a Random Forest Regression model.

### Price Prediction Techniques

Implemented a regression-based price prediction system.

### Model Evaluation

Evaluated model performance using R² Score and MAE.

### Model Deployment

Deployed the trained model using Streamlit and Joblib.

### Streamlit Application Development

Created an interactive web application for real-time predictions.

---

## Results

* R² Score: 0.9333
* MAE: ₹142,613.41
* Successfully deployed a real-time used car price prediction application.

---

## Conclusion

This project successfully demonstrates the complete machine learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model development, model evaluation, deployment, and web application development. The Random Forest model achieved strong predictive accuracy and was integrated into a Streamlit application for real-time user interaction.



