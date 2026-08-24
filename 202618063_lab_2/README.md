# mayank_ds_lab2
Assignment Information

* Course: DS605: Fundamentals of Machine Learning
* Assignment: Lab Assignment 2 – Vectorized Programming with NumPy and Data Wrangling with Pandas

* Name: Mayank Baid


* ID: 202618063


* Dataset: Kaggle Titanic Dataset (`train.csv`)



Project Details

* Part A - Vectorized Programming with NumPy
* Task 1: Generated 100 random integers (seed=42) and computed min (2), max (100), median (54.0), mean (51.54), and std dev (29.28); demonstrated `arange`, `zeros`, `ones`, `linspace`, 2D/3D slicing, and 10x10 reshaping.


* Task 2: Executed matrix addition, element-wise multiplication, matrix multiplication (`@`), transpose, determinant (10.0), inverse, and verified identity with `np.allclose()`.


* Task 3: Sampled 1,000 values from normal distribution ($\mu=50.0, \sigma=15.0$), verified sample mean (50.29) and std dev (14.68), and generated `normal_distribution.png`.




* Part B - Data Wrangling with Pandas (Titanic Dataset)
* Task 4: Loaded `train.csv` (891 rows, 12 columns), inspected metadata via `head`, `tail`, `shape`, `columns`, `info`, `describe`, and demonstrated label-based (`loc`) versus index-based (`iloc`) selection.


* Task 5: Filtered subsets using `query`: males older than 50 (47), 1st class females (94, 96.81% survived), age 20–40 with fare > median who survived (104), solo passengers under 30 who did not survive (141), and embarked S class 2/3 with fare > Southampton median (193).


* Task 6: Computed grouped aggregations for survival by `Sex`, `Pclass`, `Sex`+`Pclass`, average age/fare by `Pclass`, and metrics by `Embarked`.


* Task 7: Identified missing values (`Age`: 177 / 19.87%, `Cabin`: 687 / 77.10%, `Embarked`: 2 / 0.22%), plotted `missing_values.png`, imputed missing `Age` with mean, and detected 116 `Fare` outliers using the $1.5 \times \text{IQR}$ method.


* Task 8: Engineered `FamilySize` and `IsAlone` features; created a pivot table of survival rate across `Sex` and `Pclass`.


* Task 9: Generated `correlation_heatmap.png`, `survival_by_sex.png`, `age_vs_fare.png`, and exported `titanic_cleaned.csv`.





Key Observations

1. Gender Disparity: Females achieved a survival rate of 74.20% compared to 18.89% for males.


2. Class Hierarchy: First-class passengers had the highest survival rate, while third-class passengers had the lowest.


3. Group Extremes: First-class females had the highest overall survival rate (96.81%), whereas third-class males had the lowest (13.54%).


4. Fare Correlation: `Fare` showed a positive correlation with survival ($r = 0.26$), indicating higher-paying passengers had better survival rates.


5. Missing Data Patterns: `Age` had 19.87% missing values and was imputed with the mean, while `Cabin` had 77.10% missing data.


6. Fare Outliers: The dataset contained 116 high-end `Fare` outliers with values reaching £512.33.


7. Solitary Penalty: Solo passengers (`IsAlone = 1`) had lower survival rates than passengers travelling in small family groups.