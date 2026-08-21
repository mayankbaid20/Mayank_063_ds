#Lab Assignment - 3
Scikit-learn: Data Preprocessing and Model Performance Evaluation
#name:Mayank Baid

Dataset Link
Hotel Booking Demand Dataset (Kaggle)(https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

Preprocessing Choices

Target Distribution: Identified target variable `is_canceled` with 62.96% non-canceled (0) and 37.04% canceled (1) records.

Feature Classification: Separated dataset features into 19 numerical columns (e.g., `lead_time`, `adr`, `booking_changes`) and 12 categorical columns (e.g., `hotel`, `market_segment`, `deposit_type`).

Missing Value Handling: Identified missing values in `company` (112,593 missing / 94.31%), `agent` (16,340 missing / 13.69%), `country` (488 missing / 0.41%), and `children` (4 missing / 0.00%).


Feature Dropping: Dropped the `company` column due to extreme missingness (>94%), reducing total feature count from 31 down to 30 ($119,390 \times 30$).



Final Observations:

Target Imbalance: Moderate class imbalance (~37% cancellations) requires evaluation via F1-Score or ROC-AUC rather than standard accuracy.

Booking Source Profile: High missingness in company indicates low corporate usage, showing guests primarily book individually or via travel agencies.  

Encoding Requirements: Categorical attributes (country, market_segment) and temporal columns (reservation_status_date) require proper encoding before model training..

Lead Time Impact: Longer lead_time correlates directly with higher cancellation risk, as plans made far in advance are more prone to changes.

Deposit Type Influence: Bookings marked as Non Refund paradoxically show high cancellation rates in this dataset due to bulk institutional or agency block reservations.

Room Type Discrepancies: A change between reserved_room_type and assigned_room_type significantly lowers cancellation likelihood, as room upgrades enhance guest retention.

Special Requests & Parking: Guests requesting required_car_parking_spaces or multiple total_of_special_requests have near-zero cancellation rates, indicating strong booking intent.

Repeated Guests Behavior: is_repeated_guest shows an extremely low cancellation probability, making customer loyalty a strong predictor of completion.

Market Segment Variances: Groups and Online Travel Agents (TA) exhibit noticeably higher cancellation rates compared to Direct or Corporate segments.

