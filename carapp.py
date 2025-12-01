import streamlit as st
import pandas as pd
import joblib

# PAGE SETTINGS
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction (Random Forest - Accurate)")

# LOAD DATA
df = pd.read_csv(r"C:\Users\swetha\Downloads\DATA SCIENTIST\used_cars.csv")

df['price'] = df['price'].replace('[,$]', '', regex=True).astype(float)
df['milage'] = df['milage'].replace('[^0-9]', '', regex=True).astype(float)

# LOAD MODEL + COLUMNS
model = joblib.load("random_forest_car_price.pkl")

# Safely get model columns
try:
    model_columns = joblib.load("model_columns.pkl")
except:
    model_columns = model.feature_names_in_

# USER INPUTS
brand = st.selectbox("Brand", df['brand'].unique())
model_name = st.selectbox("Model", df['model'].unique())
model_year = st.number_input("Model Year", int(df['model_year'].min()), int(df['model_year'].max()), 2020)
milage = st.number_input("Mileage (km)", 0, int(df['milage'].max()), 40000)
fuel_type = st.selectbox("Fuel Type", df['fuel_type'].unique())
engine = st.selectbox("Engine", df['engine'].unique())
transmission = st.selectbox("Transmission", df['transmission'].unique())
ext_col = st.selectbox("Exterior Color", df['ext_col'].unique())
int_col = st.selectbox("Interior Color", df['int_col'].unique())
accident = st.selectbox("Accident History", df['accident'].unique())
clean_title = st.selectbox("Clean Title", df['clean_title'].unique())

# PREDICTION
if st.button("Predict Price 💰"):

    input_df = pd.DataFrame([{
        'brand': brand,
        'model': model_name,
        'model_year': model_year,
        'milage': milage,
        'fuel_type': fuel_type,
        'engine': engine,
        'transmission': transmission,
        'ext_col': ext_col,
        'int_col': int_col,
        'accident': accident,
        'clean_title': clean_title
    }])

    combined = pd.concat([df.drop(columns=['price']), input_df], ignore_index=True)
    combined_encoded = pd.get_dummies(combined)

    input_processed = combined_encoded.tail(1)

    input_processed = input_processed.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(input_processed)[0]
    

    # ✅ Boost prediction based on real model popularity
    model_avg_price = df.groupby('model')['price'].mean().to_dict()
    prediction += model_avg_price.get(model_name, 0) * 0.15

    st.success(f"✅ Estimated Car Price: ₹ {prediction:,.0f}")

    #  DEBUG INFO
    with st.expander("Debug Info"):
        st.write(input_processed)
        st.write("Prediction:", prediction)

    # ⭐ FEATURE IMPORTANCE (CORRECT)
    st.subheader("📊 Top Feature Importances")

    importances = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_
    ).sort_values(ascending=False)

    st.dataframe(importances.head(20))
    st.bar_chart(importances.head(15))