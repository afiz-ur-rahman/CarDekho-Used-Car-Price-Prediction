import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL & DATA
MODEL_PATH = "car_price_model.pkl"
DATA_PATH = "structured_car_data.csv"
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# PAGE CONFIG
st.set_page_config(page_title="Car Dheko - Used Car Price Prediction",layout="centered")
st.title("Car Dheko - Used Car Price Prediction")
st.write("Predict the price of a used car based on its features.")

# INPUT UI
col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox(
    "Brand",
     sorted(df["Brand"].unique())
    )

    model_name = st.selectbox(
        "Model",
        sorted(df["Model"].unique())
    )

    body_type = st.selectbox(
        "Body Type",
        sorted(df["Body_Type"].dropna().unique())
    )

    fuel = st.selectbox(
        "Fuel Type",
        sorted(df["Fuel"].unique())
    )

    transmission = st.selectbox(
        "Transmission",
        sorted(df["Transmission"].unique())
    )

with col2:
    city = st.selectbox(
        "City",
        sorted(df["City"].unique())
    )

    owner_no = st.slider(
        "Owner Number",
        int(df["Owner_No"].min()),
        int(df["Owner_No"].max()),
        int(df["Owner_No"].median())
    )

    year = st.slider(
        "Manufacturing Year",
        int(df["Year"].min()),
        int(df["Year"].max()),
        int(df["Year"].median())
    )

    km = st.number_input(
        "Kilometers Driven",
        min_value=0,
        value=int(df["KM"].median())
    )

    engine_cc = st.number_input(
        "Engine CC",
        min_value=int(df["Engine_CC"].min()),
        value=int(df["Engine_CC"].median())
    )

    seats = st.slider(
        "Seats",
        int(df["Seats"].min()),
        int(df["Seats"].max()),
        int(df["Seats"].median())
    )


if st.button("Predict Price"):
    # Create input DataFrame (ONE ROW)
    input_data = pd.DataFrame({
        "Brand": [brand],
        "Model": [model_name],
        "Body_Type": [body_type],
        "Fuel": [fuel],
        "Transmission": [transmission],
        "Owner_No": [owner_no],
        "Year": [year],
        "KM": [km],
        "Seats": [seats],
        "Engine_CC": [engine_cc],
        "City": [city]
    })

    # Ensure SAME column order as training
    input_data = input_data[model.feature_names_in_]

    # Predict
    prediction = model.predict(input_data)[0]

    # Safety: prevent negative price
    prediction = max(0, prediction)

    # Show result
    st.success(f"Estimated Price: ₹ {int(prediction):,}")
    st.balloons()

# FOOTER
st.write("---")
st.write("Developed for Car Dheko Project | Machine Learning | Streamlit Deployment")
