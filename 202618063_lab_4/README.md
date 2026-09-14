# 🏠 Airbnb Price Prediction
# name:Mayank Baid
## DS605 – Fundamentals of Machine Learning Lab Assignment 4

This project is an end-to-end machine learning project for predicting the nightly price of Airbnb listings in New York City.

The project covers data cleaning, exploratory data analysis, feature engineering, model training, model comparison, hyperparameter tuning, and deployment using Streamlit.

---

## 🚀 Live Application

The trained model is deployed as a Streamlit web application.

Live App:
https://airbnbratepredictor.streamlit.app/

The application allows users to enter Airbnb listing details and get an estimated nightly price.

---

## 📌 Project Objective

The main objective of this project is to build a machine learning model that can estimate the price of an Airbnb listing using information such as:

- Location
- Neighbourhood
- Room type
- Latitude and longitude
- Minimum nights
- Number of reviews
- Reviews per month
- Host listing count
- Availability

The final trained model is integrated into a simple Streamlit application for real-time predictions.

---

## 📂 Dataset

The project uses the New York City Airbnb Open Data 2019 dataset from Kaggle.

Dataset file: AB_NYC_2019.csv

Original dataset:
- 48,895 rows
- 16 columns

Target variable:
price

During preprocessing, duplicate records and invalid/outlier prices were removed.

After cleaning:
- 48,410 rows
- 16 columns

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Removed duplicate records.
2. Removed listings with price less than or equal to zero.
3. Removed extremely high prices above the 99th percentile.
4. Handled missing values using imputation.
5. Converted categorical variables using One-Hot Encoding.
6. Standardized numerical features using StandardScaler.

Some columns such as IDs, names, host names, and last review date were removed because they were not useful for the final prediction model.

---

## 🛠️ Feature Engineering

Additional features were created to improve the model:

- reviews_per_availability
- minimum_nights_log
- reviews_log
- availability_log

Example:

reviews_per_availability = number_of_reviews / (availability_365 + 1)

Log transformations were also used for variables with highly skewed distributions.

---

## 🤖 Machine Learning Models

Three regression models were tested:

- Linear Regression
- Ridge Regression
- Random Forest Regression

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📊 Model Results

### Initial Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 49.53 | 78.79 | 0.4295 |
| Ridge Regression | 49.47 | 78.78 | 0.4297 |
| Random Forest | 44.85 | 72.94 | 0.5111 |

Random Forest performed better than the linear models, so it was selected for further tuning.

---

## 🎯 Hyperparameter Tuning

GridSearchCV was used to tune the Random Forest model.

Parameters tested included:

- n_estimators
- max_depth
- min_samples_split

Best parameters:

- n_estimators = 200
- max_depth = 15
- min_samples_split = 5

### Final Model Performance

| Metric | Score |
|---|---:|
| MAE | 44.25 |
| RMSE | 72.36 |
| R² | 0.5189 |

The tuned Random Forest model was selected as the final model.

---

## 🔍 Important Features

Some of the most important features in the final Random Forest model were:

| Feature | Importance |
|---|---:|
| Entire home/apt | 0.3487 |
| Longitude | 0.1561 |
| Latitude | 0.1160 |
| Reviews / Availability | 0.0687 |
| Host Listings Count | 0.0452 |
| Reviews per Month | 0.0407 |

The results show that room type and location have a strong influence on Airbnb prices.

---

## 🌐 Streamlit Application

The final model was saved as a complete preprocessing and prediction pipeline and integrated into a Streamlit application.

The application allows users to enter:

- Neighbourhood group
- Neighbourhood
- Room type
- Latitude
- Longitude
- Minimum nights
- Number of reviews
- Reviews per month
- Host listing count
- Availability

After entering the details, the application displays the estimated nightly Airbnb price.

### Application Screenshot

Add the Streamlit screenshot here after uploading it to the screenshots folder:

screenshots/streamlit_app.png

---

## 📁 Project Structure

202618063_lab_4/
|
|-- app/
|   |-- airbnb_predictor.py
|
|-- models/
|   |-- airbnb_price_pipeline.pkl
|
|-- plots/
|   |-- price_distribution.png
|   |-- median_price_by_room_type.png
|   |-- model_comparison.png
|   |-- feature_importance.png
|
|-- screenshots/
|   |-- streamlit_app.png
|
|-- AB_NYC_2019.csv
|-- analysis.ipynb
|-- README.md
|-- requirements.txt

---

## 💻 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## ▶️ How to Run the Project Locally

### 1. Clone the repository

git clone https://github.com/mayankbaid20/Mayank_063_ds.git

### 2. Go to the Lab 4 folder

cd Mayank_063_ds/202618063_lab_4

### 3. Install the required libraries

pip install -r requirements.txt

### 4. Run the Streamlit application

python -m streamlit run app/airbnb_predictor.py

The application will open in your browser.

---

## 📓 Jupyter Notebook

The complete machine learning workflow is available in:

analysis.ipynb

The notebook includes:

- Dataset loading
- Data inspection
- Data cleaning
- Exploratory data analysis
- Feature engineering
- Preprocessing
- Model training
- Model comparison
- Hyperparameter tuning
- Final model evaluation
- Feature importance
- Model saving

---

## 💾 Saved Model

The final trained model is saved as:

models/airbnb_price_pipeline.pkl

The saved pipeline contains both the preprocessing steps and the trained Random Forest model.

This allows the same preprocessing used during training to be applied when making predictions through the Streamlit application.

---

## 📈 Key Findings

Some useful observations from the analysis were:

- Entire home/apt listings generally have higher prices than private and shared rooms.
- Location has a strong effect on Airbnb prices.
- Manhattan and other neighbourhood groups show noticeable price differences.
- Random Forest performed better than Linear Regression and Ridge Regression.
- Feature engineering and preprocessing helped prepare the data for machine learning.
- Hyperparameter tuning slightly improved the Random Forest performance.

---

## ⚠️ Limitations

There are some limitations to this project:

- The dataset is from 2019, so predicted prices may not represent current Airbnb prices.
- The dataset does not contain detailed information about amenities and some property-specific factors.
- Seasonal changes and current market demand are not included.
- The model provides an estimate and should not be treated as an exact market price.
- Predictions can vary depending on the quality and range of the input data.

---

## 🎓 Conclusion

This project demonstrates a complete machine learning workflow, starting from raw Airbnb data and ending with a deployed prediction application.

The tuned Random Forest Regressor achieved an R² score of 0.5189 on the test set with an RMSE of 72.36.

The final model was saved as a reusable pipeline and deployed through Streamlit so that users can interact with the model and receive Airbnb price estimates.

---

## 👨‍🎓 Project Information

Course: DS605 – Fundamentals of Machine Learning

Assignment: Lab Assignment 4 – End-to-End Machine Learning Project

Project: Airbnb Price Prediction

Dataset: New York City Airbnb Open Data 2019

Model: Tuned Random Forest Regressor

Deployment: Streamlit Community Cloud

---


