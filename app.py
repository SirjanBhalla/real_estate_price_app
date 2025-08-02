import streamlit as st
import joblib
import numpy as np
from real_estate_price import config


model = joblib.load(config.MODEL_SAVE_PATH)


st.set_page_config(page_title="Real Estate Price Predictor", page_icon=":house:", layout="centered")

st.title("🏠 Real Estate Price Predictor")
st.markdown(
    "Enter the property details below to get an estimated sale price.\n\n*Built with care by an Algonquin Student*"
)
st.divider()

st.markdown("### Please provide the following property details:")


col1, col2, col3 = st.columns(3)

with col1:
    year_sold = st.number_input(
        "Year Sold",
        min_value=1940,
        max_value=2025,
        step=1,
        help="Year the transaction took place."
    )
    property_tax = st.number_input(
        "Monthly Property Tax (USD)",
        min_value=0,
        step=1,
        help="Monthly property tax cost in US dollars."
    )
    insurance = st.number_input(
        "Monthly Homeowner's Insurance (USD)",
        min_value=0,
        step=1,
        help="Monthly insurance cost in US dollars."
    )
    beds = st.number_input(
        "Number of Bedrooms",
        min_value=0,
        step=1,
        help="Total number of bedrooms in the property."
    )
    baths = st.number_input(
        "Number of Bathrooms",
        min_value=0,
        step=1,
        help="Total number of bathrooms in the property."
    )

with col2:
    sqft = st.number_input(
        "Total Floor Area (sq ft)",
        min_value=0,
        step=1,
        help="Total living area in square feet."
    )
    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2025,
        step=1,
        help="Year when the property was constructed."
    )
    lot_size = st.number_input(
        "Lot Size (sq ft)",
        min_value=0,
        step=1,
        help="Total outside area in square feet."
    )
    basement_str = st.radio(
        "Basement",
        options=["Yes", "No"],
        index=1,
        help="Does the property have a basement?"
    )
    popular_str = st.radio(
        "Is this a popular property? (2 beds & 2 baths)",
        options=["Yes", "No"],
        index=1,
        help="Based on feature engineering: set to Yes if the property has 2 bedrooms and 2 bathrooms."
    )

with col3:
    recession_str = st.radio(
        "Was this home sold during 2010–2013 recession?",
        options=["Yes", "No"],
        index=1,
        help="Indicates if the home was sold during an economic recession affecting prices."
    )
    property_type_label = st.selectbox(
        "Property Type",
        options=["Bungalow", "Condo"],
        index=1,
        help="Select 'Condo' or 'Bungalow'."
    )

st.divider()

if st.button("Predict Property Price"):

    basement = 1 if basement_str == "Yes" else 0
    popular = 1 if popular_str == "Yes" else 0
    recession = 1 if recession_str == "Yes" else 0
    property_type_Condo = 1 if property_type_label == "Condo" else 0


    X = np.array([[
        year_sold,
        property_tax,
        insurance,
        beds,
        baths,
        sqft,
        year_built,
        lot_size,
        basement,
        popular,
        recession,
        property_type_Condo
    ]])


    prediction = model.predict(X)[0]


    price_formatted = f"${prediction:,.0f}"


    st.success(f"Estimated Property Price: {price_formatted}")

    st.info(
        "Note: This is an estimate based on available historical data and model predictions. Actual prices may vary."
    )
