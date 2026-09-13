import streamlit as st
import pandas as pd
import numpy as np
import joblib


# -----------------------------------
# Page settings
# -----------------------------------

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# -----------------------------------
# Custom styling
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #fff5f7 0%,
        #f7f3ff 50%,
        #eef8ff 100%
    );
}

/* Main title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #ff385c;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555555;
    margin-bottom: 30px;
}


/* Information cards */
.info-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.08);
}

.info-number {
    font-size: 28px;
    font-weight: 700;
    color: #ff385c;
}

.info-text {
    font-size: 14px;
    color: #666666;
}


/* Section heading */
.section-title {
    font-size: 24px;
    font-weight: 600;
    color: #333333;
    margin-top: 25px;
    margin-bottom: 15px;
}


/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #ff385c,
        #ff6b81
    );
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-size: 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #e61e4d,
        #ff385c
    );
    color: white;
}


/* Prediction result */
.result-box {
    background: linear-gradient(
        135deg,
        #ff385c,
        #ff6b81
    );
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-top: 25px;
    box-shadow: 0px 6px 20px rgba(255, 56, 92, 0.30);
}

.result-title {
    font-size: 20px;
    margin-bottom: 10px;
}

.result-price {
    font-size: 44px;
    font-weight: 700;
    margin-bottom: 8px;
}

.result-description {
    font-size: 15px;
}


/* Footer */
.footer {
    text-align: center;
    color: #777777;
    margin-top: 35px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# Load trained model
# -----------------------------------

model = joblib.load(
    "../models/airbnb_price_pipeline.pkl"
)


# -----------------------------------
# Header
# -----------------------------------

st.markdown(
    '<div class="title">🏠 Airbnb Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate the nightly price of an Airbnb listing in New York City'
    '</div>',
    unsafe_allow_html=True
)


# -----------------------------------
# Information cards
# -----------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">NYC</div>
        <div class="info-text">Location</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">2019</div>
        <div class="info-text">Dataset Year</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">ML</div>
        <div class="info-text">Prediction Model</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------
# Listing details
# -----------------------------------

st.markdown(
    '<div class="section-title">📍 Listing Details</div>',
    unsafe_allow_html=True
)


# -----------------------------------
# Input fields
# -----------------------------------

col1, col2 = st.columns(2)


with col1:

    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        [
            "Bronx",
            "Brooklyn",
            "Manhattan",
            "Queens",
            "Staten Island"
        ]
    )

    neighbourhood = st.text_input(
        "Neighbourhood",
        "Harlem"
    )

    room_type = st.selectbox(
        "Room Type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )

    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        value=3
    )

    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        value=10
    )


with col2:

    latitude = st.number_input(
        "Latitude",
        value=40.72,
        format="%.4f"
    )

    longitude = st.number_input(
        "Longitude",
        value=-73.95,
        format="%.4f"
    )

    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0,
        value=1.0
    )

    calculated_host_listings_count = st.number_input(
        "Host Listings Count",
        min_value=0,
        value=1
    )

    availability_365 = st.number_input(
        "Availability (365 days)",
        min_value=0,
        max_value=365,
        value=200
    )


st.write("")


# -----------------------------------
# Prediction button
# -----------------------------------

if st.button("💰 Predict Airbnb Price"):

    # Feature engineering
    reviews_per_availability = (
        number_of_reviews /
        (availability_365 + 1)
    )

    minimum_nights_log = np.log1p(
        minimum_nights
    )

    reviews_log = np.log1p(
        number_of_reviews
    )

    availability_log = np.log1p(
        availability_365
    )


    # Create input dataframe
    input_data = pd.DataFrame({
        "neighbourhood_group": [
            neighbourhood_group
        ],

        "neighbourhood": [
            neighbourhood
        ],

        "latitude": [
            latitude
        ],

        "longitude": [
            longitude
        ],

        "room_type": [
            room_type
        ],

        "minimum_nights": [
            minimum_nights
        ],

        "number_of_reviews": [
            number_of_reviews
        ],

        "reviews_per_month": [
            reviews_per_month
        ],

        "calculated_host_listings_count": [
            calculated_host_listings_count
        ],

        "availability_365": [
            availability_365
        ],

        "reviews_per_availability": [
            reviews_per_availability
        ],

        "minimum_nights_log": [
            minimum_nights_log
        ],

        "reviews_log": [
            reviews_log
        ],

        "availability_log": [
            availability_log
        ]
    })


    # -----------------------------------
    # Make prediction
    # -----------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # -----------------------------------
    # Display result
    # -----------------------------------

    st.markdown(
        f"""<div class="result-box">
<div class="result-title">Estimated Nightly Price</div>
<div class="result-price">${prediction:,.2f}</div>
<div class="result-description">
Based on the information provided
</div>
</div>""",
        unsafe_allow_html=True
    )


# -----------------------------------
# Footer
# -----------------------------------

st.markdown(
    '<div class="footer">'
    'DS605 - Fundamentals of Machine Learning | '
    'Airbnb Price Prediction Project'
    '</div>',
    unsafe_allow_html=True
)