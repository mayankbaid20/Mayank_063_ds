# DS605 Lab Assignment 5: Regression & Classification, Scikit-learn vs. From Scratch

Assignment Information

* Course: DS605: Fundamentals of Machine Learning
* Assignment: Lab Assignment 5 – Regression and Classification, Scikit-learn vs. From-Scratch NumPy/Pandas Implementations
* Name: *\<Mayank Baid\>*
* ID: *\<202618063\>*
* Dataset: Garments Worker Productivity Dataset (`garments_worker_productivity.csv`)

Project Details

* Part A – Data Preparation
  * Task 1: Loaded `garments_worker_productivity.csv` (1,197 rows, 15 columns), stripped whitespace from column names, and inspected shape, dtypes, and summary statistics via `info()` and `describe()`.
  * Task 2: Checked for missing values and found `wip` (work-in-progress) missing in 506 rows (42.3%); dropped the `date` column and any fully-empty columns.
  * Task 3: Engineered the classification target `MeetsTarget` (1 if `actual_productivity >= targeted_productivity`, else 0) and confirmed the resulting class balance.
  * Task 4: Defined two separate feature sets — one for regression (target: `actual_productivity`) and one for classification (target: `MeetsTarget`) — with `actual_productivity` explicitly excluded from the classification features to prevent target leakage.
  * Task 5: Created a single fixed 80/20 train-test split (`random_state=42`, 957 training rows / 240 test rows) and reused the same indices across every implementation so all models are evaluated on identical data.
  * Task 6: Identified numeric versus categorical columns (`department`, `day`, `quarter` are categorical; the rest are numeric) to drive the two separate preprocessing pipelines below.

* Part B – Scikit-learn Implementation
  * Task 7: Built a `ColumnTransformer` pipeline — median imputation and `StandardScaler` for numeric columns, most-frequent imputation and `OneHotEncoder` for categorical columns — and fit it on the training split only.
  * Task 8: Trained `LinearRegression` on the processed features; achieved MAE = 0.1085, RMSE = 0.1486, R² = 0.1682, with training time ≈ 0.0138s and prediction time ≈ 0.00075s.
  * Task 9: Trained `LogisticRegression` (`max_iter=2000`) on the processed features; achieved accuracy = 77.5%, precision = 78.08%, recall = 96.61%, F1 = 86.36%, with training time ≈ 0.1203s and prediction time ≈ 0.00035s.

* Part C – From-Scratch Implementation (NumPy + Pandas only)
  * Task 10: Replicated preprocessing manually — median/mode imputation with pandas, one-hot encoding via `pd.get_dummies`, and standardization using training-set mean and standard deviation only (never touching test-set statistics).
  * Task 11: Implemented Linear Regression from the closed-form normal equation, β = (XᵀX)⁻¹Xᵀy, using `np.linalg.pinv`; achieved MAE = 0.1085, RMSE = 0.1486, R² = 0.1682 (matching scikit-learn to five decimal places), with training time ≈ 0.0935s and prediction time ≈ 0.00004s.
  * Task 12: Implemented Logistic Regression from scratch with a hand-written sigmoid function and batch gradient descent (learning rate = 0.05, up to 10,000 epochs, tolerance-based early stopping); achieved accuracy = 77.5%, precision = 78.34%, recall = 96.05%, F1 = 86.29%, with training time ≈ 0.3631s and prediction time ≈ 0.00018s.
  * Task 13: Hand-wrote every evaluation metric (MAE, RMSE, R² for regression; accuracy, precision, recall, F1 for classification) from their base formulas and cross-checked them against `sklearn.metrics` to confirm the manual implementations are numerically correct.

* Part D – Optimized From-Scratch Classification
  * Task 14: Rebuilt the logistic regression training loop with fully vectorized gradient updates plus L2 regularization (learning rate = 0.03, up to 20,000 epochs, regularization strength = 0.001).
  * Task 15: Evaluated the optimized model; achieved accuracy = 77.5%, precision = 78.34%, recall = 96.05%, F1 = 86.29% — identical to the unregularized from-scratch model, with training time ≈ 0.6551s and prediction time ≈ 0.00019s.

* Part E – Results & Export
  * Task 16: Assembled `regression_comparison.csv` (Scikit-learn vs. From Scratch) and `classification_comparison.csv` (Scikit-learn vs. From Scratch vs. Optimized From Scratch), each including accuracy/error metrics alongside training and prediction time.
  * Task 17: Exported `final_results.csv`, flattening every metric from every implementation into one file for easy lookup.
  * Task 18: Ran a final verification cell printing dataset shape, split sizes, and every model's headline metrics side by side as a sanity check before submission.

Key Observations

1. Implementation Parity: The from-scratch linear regression matches scikit-learn's MAE, RMSE, and R² to five decimal places, confirming the closed-form normal-equation implementation is mathematically correct.
2. Classification Agreement: All three classification implementations converge to the same 77.5% accuracy, with precision and recall differing by less than half a percentage point between the scikit-learn and manual versions.
3. Regularization Had Little Effect: The L2-regularized optimized model produced results identical to the unregularized from-scratch model, suggesting overfitting was not a significant concern at this dataset size (1,197 rows).
4. Training Speed: Scikit-learn's regression pipeline trains roughly 7x faster than the from-scratch normal-equation solve at this scale, largely due to optimized linear algebra routines under the hood.
5. Prediction Speed: The from-scratch models predict faster than the scikit-learn pipeline once trained, since they skip the pipeline-transform overhead at inference time.
6. Missing Data: The `wip` column was missing in 42.3% of rows, the largest gap in the dataset, and was imputed with the median rather than dropped to preserve sample size.
7. Modest Predictive Power: An R² of 0.168 indicates the available features explain only a portion of the variance in `actual_productivity` — factors like individual worker skill or machine condition, which aren't in the dataset, likely account for the rest.
8. No Target Leakage: `actual_productivity` was deliberately excluded from the classification feature set, since it was used to construct the `MeetsTarget` label itself.
