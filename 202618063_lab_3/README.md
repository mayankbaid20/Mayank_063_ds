# Lab Assignment - 3

## Scikit-learn: Data Preprocessing and Model Performance Evaluation

**Name:** Mayank Baid
**Course:** DS605: Fundamentals of Machine Learning

## Dataset

**Dataset:** Hotel Booking Demand Dataset
**Source:** Kaggle – Jesse Mostipak
**Dataset Link:** https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

The dataset contains **119,390 hotel booking records**. The target variable is `is_canceled`, which indicates whether a booking was canceled.

## Preprocessing & Workflow

### 1. Target Analysis

* `0` – Not Canceled: **62.96%**
* `1` – Canceled: **37.04%**
* The moderate class imbalance makes **F1-Score and ROC-AUC** more useful than accuracy alone.

### 2. Feature Classification

* **19 numerical features:** `lead_time`, `adr`, `booking_changes`, etc.
* **12 categorical features:** `hotel`, `market_segment`, `deposit_type`, etc.

### 3. Missing Values

Missing values were identified in:

* `company`: **112,593 (94.31%)**
* `agent`: **16,340 (13.69%)**
* `country`: **488 (0.41%)**
* `children`: **4**

The `company` column was dropped due to its extremely high missingness, reducing the dataset to **119,390 × 30**.

### 4. Encoding & Preprocessing

Categorical variables such as `country`, `market_segment`, `hotel`, and `deposit_type` require encoding before model training. Temporal data such as `reservation_status_date` also requires suitable feature transformation.

## Key Insights

* **Lead Time:** Longer lead times are generally associated with higher cancellation risk.
* **Deposit Type:** `Non Refund` bookings show unexpectedly high cancellation rates in this dataset.
* **Room Type:** Differences between reserved and assigned room types are associated with lower cancellation likelihood.
* **Special Requests:** Parking requirements and multiple special requests indicate stronger booking intent and lower cancellation rates.
* **Repeated Guests:** Repeated guests have extremely low cancellation probability.
* **Market Segment:** Groups and Online Travel Agents show higher cancellation rates than Direct and Corporate segments.
* **Company Missingness:** The high missingness in `company` suggests limited corporate booking information.

## Model Evaluation

Due to the class imbalance, model performance should be evaluated using:

* **Precision**
* **Recall**
* **F1-Score**
* **ROC-AUC**
* Accuracy as a supplementary metric

## Project Structure

```text
Mayank_063_ds/
│
├── preprocessing.ipynb
├── hotel_bookings.csv
└── README.md
```

## Conclusion

The preprocessing workflow prepares the Hotel Booking Demand dataset for machine learning by handling missing values, classifying features, identifying encoding requirements, and selecting appropriate evaluation metrics. The analysis highlights **lead time, deposit type, market segment, repeated guest status, and special requests** as important factors related to booking cancellation.
