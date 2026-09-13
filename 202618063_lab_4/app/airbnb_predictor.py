import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #fff5f7 0%,
            #f8f4ff 50%,
            #eef8ff 100%
        );
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #ff385c;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555555;
        margin-bottom: 30px;
    }

    .info-card {
        background: white;
        padding: 18px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }

    .info-title {
        font-size: 16px;
        font-weight: 700;
        color: #333333;
    }

    .info-text {
        font-size: 14px;
        color: #777777;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #333333;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .result-box {
        background: white;
        padding: 28px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.10);
        border: 2px solid #ff385c;
        margin-top: 25px;
        margin-bottom: 20px;
    }

    .result-title {
        font-size: 20px;
        font-weight: 700;
        color: #444444;
    }

    .result-price {
        font-size: 42px;
        font-weight: 800;
        color: #ff385c;
        margin-top: 8px;
    }

    .result-description {
        font-size: 14px;
        color: #777777;
        margin-top: 5px;
    }

    .footer {
        text-align: center;
        color: #777777;
        font-size: 13px;
        margin-top: 35px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🏠 Airbnb Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Estimate the nightly price of an Airbnb listing in New York City</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Information cards
# --------------------------------------------------

card1, card2, card3 = st.columns(3)

with card1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">📍 Location</div>
            <div class="info-text">New York City</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with card2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">📅 Dataset</div>
            <div class="info-text">Airbnb NYC 2019</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with card3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">🤖 Model</div>
            <div class="info-text">Random Forest Regression</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Load saved model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "airbnb_price_pipeline.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(
        "Could not load the trained model. "
        "Please make sure 'airbnb_price_pipeline.pkl' "
        "is inside the 'models' folder."
    )
    st.stop()


# --------------------------------------------------
# Input section
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📝 Enter Airbnb Details</div>',
    unsafe_allow_html=True
)


left_col, right_col = st.columns(2)


# --------------------------------------------------
# Left column inputs
# --------------------------------------------------

with left_col:

    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        [
            "Manhattan",
            "Brooklyn",
            "Queens",
            "Bronx",
            "Staten Island"
        ]
    )

    room_type = st.selectbox(
        "Room Type",
        [
            "Entire home/apt",
            "Private room",
            "Shared room"
        ]
    )

    latitude = st.number_input(
        "Latitude",
        min_value=40.49,
        max_value=40.92,
        value=40.72,
        format="%.6f"
    )

    longitude = st.number_input(
        "Longitude",
        min_value=-74.30,
        max_value=-73.65,
        value=-73.99,
        format="%.6f"
    )

    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        max_value=365,
        value=3,
        step=1
    )

    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        max_value=1000,
        value=20,
        step=1
    )


# --------------------------------------------------
# Right column inputs
# --------------------------------------------------

with right_col:

    neighbourhood = st.text_input(
        "Neighbourhood",
        value="Midtown"
    )

    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0,
        max_value=100.0,
        value=1.5,
        step=0.1
    )

    calculated_host_listings_count = st.number_input(
        "Host Listings Count",
        min_value=1,
        max_value=1000,
        value=1,
        step=1
    )

    availability_365 = st.number_input(
        "Availability (365 days)",
        min_value=0,
        max_value=365,
        value=200,
        step=1
    )


# --------------------------------------------------
# Prediction button
# --------------------------------------------------

st.write("")

predict_button = st.button(
    "💰 Predict Airbnb Price",
    use_container_width=True
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    # Feature engineering
    reviews_per_availability = (
        number_of_reviews / (availability_365 + 1)
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
    input_data = pd.DataFrame(
        {
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
        }
    )


    # Make prediction
    try:

        prediction = model.predict(input_data)[0]

        # Avoid negative predicted prices
        prediction = max(0, prediction)


        # Display result
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

    except Exception as e:

        st.error(
            "Prediction could not be generated. "
            "Please check the entered values and model file."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        Airbnb Price Prediction • Machine Learning Lab Project<br>
        Random Forest Regression trained on NYC Airbnb 2019 data
    </div>
    """,
    unsafe_allow_html=True
)