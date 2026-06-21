Car Dheko - Used Car Price Prediction
Project Overview

This project focuses on predicting the selling price of used cars using Machine Learning techniques. A Random Forest Regression model was developed to estimate car prices based on vehicle specifications such as brand, model, fuel type, transmission, manufacturing year, kilometers driven, seating capacity, engine capacity, ownership history, and city.

The trained model was deployed using Streamlit to provide real-time price predictions through a user-friendly web application.

Problem Statement

Determining the fair market value of a used car is often challenging due to multiple influencing factors. The objective of this project is to develop a machine learning solution that can accurately predict used car prices and assist users in making informed buying and selling decisions.

Dataset Information
Total Records: 8,369
Target Variable: Price
Features Used
Brand
Model
Body_Type
Fuel
Transmission
Owner_No
Year
KM
Seats
Engine_CC
City
Data Preprocessing

The following preprocessing techniques were applied:

Missing value handling using SimpleImputer
Categorical feature encoding using OneHotEncoder
Feature selection and preparation
Data transformation using Scikit-Learn Pipelines
Exploratory Data Analysis (EDA)

EDA was performed to understand:

Brand-wise distribution
Fuel type distribution
Transmission analysis
Manufacturing year trends
Kilometers driven analysis
City-wise vehicle distribution
Price distribution patterns
Machine Learning Model
Algorithm Used
Random Forest Regressor
Model Parameters
n_estimators = 300
random_state = 42
n_jobs = -1
Model Evaluation

The model was evaluated using standard regression metrics.

Results
R² Score: 0.9333
MAE (Mean Absolute Error): ₹142,613.41

The model achieved strong predictive performance and accurately estimated used car prices.

Model Deployment

The trained model was saved using Joblib and deployed through a Streamlit application.

Application Features
Brand Selection
Model Selection
Fuel Type Selection
Transmission Selection
Manufacturing Year Input
Kilometers Driven Input
Engine Capacity Input
Seating Capacity Input
City Selection
Instant Price Prediction
Technologies Used
Python
Pandas
NumPy
Scikit-Learn
Joblib
Streamlit
Skills Demonstrated
Data Cleaning and Preprocessing
Exploratory Data Analysis (EDA)
Feature Engineering
Machine Learning Model Development
Random Forest Regression
Price Prediction Techniques
Model Evaluation
Model Optimization
Model Deployment
Streamlit Application Development
Documentation and Reporting
Conclusion

This project successfully developed an end-to-end machine learning solution for used car price prediction. The Random Forest model achieved an R² Score of 0.9333 and was deployed using Streamlit, enabling users to obtain real-time car price estimates through an interactive web interface.
