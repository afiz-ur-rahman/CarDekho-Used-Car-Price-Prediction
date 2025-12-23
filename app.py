import streamlit as st
import pandas as pd
import joblib

# ================================
# LOAD MODEL & DATA
# ================================
MODEL_PATH = "car_price_model.pkl"
DATA_PATH = "structured_car_data.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

st.set_page_config(page_title="Car Dheko - Used Car Price Prediction",
                   layout="centered")

st.title("🚗 Car Dheko - Used Car Price Prediction")
st.write("Predict the price of a used car based on its features.")


# ================================
# INPUT UI
# ================================
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Select Brand", sorted(df["Brand"].unique()))
    model_name = st.selectbox("Select Model", sorted(df["Model"].unique()))
    body_type = st.selectbox("Body Type", sorted(df["Body_Type"].dropna().unique()))
    fuel = st.selectbox("Fuel Type", sorted(df["Fuel"].unique()))
    transmission = st.selectbox("Transmission Type", sorted(df["Transmission"].unique()))

with col2:
    city = st.selectbox("City", sorted(df["City"].unique()))
    owner_no = st.slider("Owner Number", int(df["Owner_No"].min()), int(df["Owner_No"].max()), 1)
    year = st.slider("Manufacturing Year", 2000, 2023, 2018)
    km = st.number_input("Kilometers Driven", min_value=0, value=30000)
    engine_cc = st.number_input("Engine CC", min_value=600, value=1200)
    seats = st.slider("Seats", 2, 10, 5)


# ================================
# PREDICTION
# ================================
if st.button("Predict Price"):
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

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")
    st.balloons()


# ================================
# FOOTER
# ================================
st.write("---")
st.write("Developed for Car Dheko Project | Machine Learning | Streamlit Deployment")
