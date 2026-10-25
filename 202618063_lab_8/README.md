# DS605: Fundamentals of Machine Learning
## Lab Assignment 8 – Decision Trees and Support Vector Machines

**Name:** Mayank Baid  
**ID:** 202618063 
**Dataset:** UCI Banknote Authentication Dataset  


## Project Overview

This project focuses on implementing Decision Tree and Linear Support Vector Machine (SVM) algorithms from scratch and comparing their performance with Scikit-learn implementations.

The UCI Banknote Authentication dataset is used to classify banknotes as genuine or forged based on four numerical features extracted from images of banknotes.

The project explores decision-tree splitting criteria, recursive tree construction, linear SVM optimization using hinge loss, margin visualization, and nonlinear SVM kernels. It also evaluates different classifiers using standard performance metrics and computational timings.

## Pipeline Architecture & Workflow

### 1. Part A: Decision Tree from Scratch

- **Dataset Preparation:** Loaded the UCI Banknote Authentication dataset using Pandas.
- **Impurity Measures:** Implemented Gini impurity and Entropy using NumPy.
- **Feature Splitting:** Evaluated feature thresholds and selected the best split using information gain (impurity reduction).
- **Recursive Tree Construction:** Built a decision tree recursively using feature-based splitting.
- **Stopping Conditions:** Implemented stopping rules based on maximum depth, pure nodes, and minimum samples required for splitting.
- **Prediction:** Developed a custom prediction function that traverses the learned tree to classify new samples.
- **Visualization:** Plotted decision boundaries using two features:
  - Variance
  - Skewness
- **Scikit-learn Comparison:** Trained a Scikit-learn DecisionTreeClassifier using the same two features and a maximum depth of 3.
- **Tree Interpretation:** Printed the learned manual decision tree structure and its feature-threshold rules.

**Learned Manual Decision Tree (Depth = 3):**

```text
[variance <= 0.7590]
  [skewness <= 5.1401]
    [variance <= 0.3180]
      Predict Class: 1
      Predict Class: 1
    [variance <= -4.1479]
      Predict Class: 1
      Predict Class: 0
  [variance <= 1.7875]
    [skewness <= 4.8400]
      Predict Class: 0
      Predict Class: 0
    [variance <= 2.3917]
      Predict Class: 0
      Predict Class: 0
```

The decision-boundary plots illustrate how the manually implemented tree and Scikit-learn tree divide the feature space into classification regions.

### 2. Part B: Linear SVM from Scratch

- **Feature Scaling:** Implemented a custom standardization function using training-set mean and standard deviation.
- **Label Transformation:** Converted binary class labels into -1 and +1 for SVM optimization.
- **Hinge Loss:** Implemented the linear soft-margin SVM optimization procedure using hinge-loss-based subgradient updates.
- **Gradient/Subgradient Descent:** Updated the weight vector and bias iteratively using NumPy.
- **Regularization:** Incorporated the regularization parameter C to control the trade-off between margin maximization and classification errors.
- **Prediction:** Implemented a custom decision function and prediction method.
- **Margin Visualization:** Plotted:
  - Separating hyperplane: \(w^T x + b = 0\)
  - Positive and negative margin boundaries: \(w^T x + b = +1\) and \(w^T x + b = -1\)
  - Margin-violating samples with green circular outlines.
- **Scikit-learn Comparison:** Compared the manual implementation with Scikit-learn LinearSVC using the same scaled two-feature representation.

**Model Configuration:**
- C = 1.0
- Learning rate = 0.01
- Iterations = 2,000

The visualization demonstrates the separating hyperplane, margin boundaries, and samples that violate the SVM margin.

### 3. Part C: Kernel SVM and Nonlinear Decision Boundaries

- **Dataset Generation:** Generated a synthetic nonlinear two-dimensional dataset using Scikit-learn's `make_moons`.
- **Dataset Configuration:**
  - Number of samples: 300
  - Noise: 0.2
  - Random state: 42
- **Kernel Implementation:** Trained Scikit-learn SVM classifiers using three different kernels:
  - Linear Kernel
  - Polynomial (Poly) Kernel
  - Radial Basis Function (RBF) Kernel
- **Decision Boundary Visualization:** Plotted the classification regions for all three kernels.
- **Comparison:** Observed how different kernels produce different decision boundaries for nonlinear data.

**Key Observation:**

- Linear kernels generate straight decision boundaries and are suitable for approximately linearly separable data.
- Polynomial kernels introduce nonlinear decision boundaries using polynomial feature interactions.
- RBF kernels can capture more complex nonlinear patterns by measuring similarity between data points.

The experiment demonstrates why nonlinear kernels can be more effective when a straight-line decision boundary cannot adequately separate the classes.

### 4. Part D: Final Model Comparison

For the final evaluation, all four original Banknote Authentication features were used:

1. Variance
2. Skewness
3. Curtosis
4. Entropy

**Dataset Preparation:**
- Total samples: 1,372
- Input features: 4
- Classes: 2
- Training samples: 1,097 (80%)
- Testing samples: 275 (20%)
- Train-test split: Stratified
- Random state: 42

The following five models were implemented and evaluated:

- Manual Decision Tree (Gini, maximum depth = 5)
- Scikit-learn Decision Tree (Gini, maximum depth = 5)
- Manual Linear SVM (C = 1.0, learning rate = 0.01, 2,000 iterations)
- Scikit-learn Linear SVM (LinearSVC, C = 1.0)
- Scikit-learn RBF SVM (C = 1.0)

Standard scaling was applied to the SVM inputs, with scaling parameters calculated using the training data.

**Performance Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Training Time (milliseconds)
- Prediction Time (milliseconds)

**Final Model Performance Comparison:**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Manual Decision Tree | 98.18% | 98.35% | 97.54% | 97.94% |
| Scikit-learn Decision Tree | 99.27% | 98.39% | 100% | 99.19% |
| Manual Linear SVM | 86.91% | 89.81% | 79.51% | 84.35% |
| Scikit-learn Linear SVM | 98.55% | 96.83% | 100% | 98.39% |
| Scikit-learn RBF SVM | 100% | 100% | 100% | 100% |

**Computational Time Comparison:**

| Model | Training Time (ms) | Prediction Time (ms) |
|---|---:|---:|
| Manual Decision Tree | 669.69 | 0.44 |
| Scikit-learn Decision Tree | 2.33 | 0.20 |
| Manual Linear SVM | 23,965.93 | 0.04 |
| Scikit-learn Linear SVM | 1.37 | 0.21 |
| Scikit-learn RBF SVM | 4.87 | 1.65 |

## Key Insights and Observations

- **Decision Tree Performance:** The manual Decision Tree achieved 98.18% accuracy, while the Scikit-learn implementation achieved 99.27%.
- **Interpretability:** Decision Trees provide understandable feature-threshold rules, making their classification decisions easier to interpret.
- **Manual SVM Performance:** The manually implemented Linear SVM achieved 86.91% accuracy, showing a noticeable performance gap compared with Scikit-learn LinearSVC.
- **Optimization Efficiency:** The manual SVM required considerably more training time because of its iterative sample-by-sample subgradient updates.
- **Linear vs Nonlinear Kernels:** The kernel experiment demonstrates that nonlinear kernels, particularly RBF, can capture complex decision boundaries that a linear kernel cannot represent.
- **Best Performing Model:** The Scikit-learn RBF SVM achieved 100% accuracy, precision, recall, and F1-score on the held-out test set.
- **Training Efficiency:** Scikit-learn implementations were substantially faster than the manually implemented algorithms in this experiment.
- **Feature Scaling:** Scaling is important for SVM models because features with different numerical ranges can influence the optimization process.

Overall, the experiments demonstrate the differences between implementing algorithms from scratch and using optimized machine-learning libraries, particularly in terms of computational efficiency and predictive performance.

## Important Implementation Details

- Gini impurity and Entropy are implemented manually using NumPy.
- Decision Tree construction, feature selection, threshold evaluation, and prediction are implemented without Scikit-learn.
- The Linear SVM optimization procedure uses NumPy-based subgradient updates and hinge-loss conditions.
- Feature scaling is implemented manually using training-set statistics.
- Two features (Variance and Skewness) are used for 2D decision-boundary and margin visualizations.
- All four original features are used for final performance evaluation.
- Scikit-learn is used for comparison models, nonlinear SVM kernels, dataset generation, evaluation metrics, and visualization support.
- A consistent stratified 80:20 train-test split is maintained for the final model comparison.

## Project Structure

```text
Mayank_063_ds/
│
├── notebook.ipynb
│
├── decision_tree_boundaries.png
│
├── svm_boundaries.png
│
├── kernel_svm_comparison.png
│
├── README.md
│
└── data/
    └── data_banknote_authentication.txt
```

### File Description

- **notebook.ipynb:** Complete implementation of manual Decision Tree, manual Linear SVM, Scikit-learn comparisons, kernel experiments, visualizations, and final performance evaluation.
- **decision_tree_boundaries.png:** Comparison of manual and Scikit-learn Decision Tree decision boundaries.
- **svm_boundaries.png:** Manual Linear SVM separating hyperplane, margins, margin violators, and Scikit-learn LinearSVC boundary.
- **kernel_svm_comparison.png:** Comparison of Linear, Polynomial, and RBF SVM decision boundaries.
- **data_banknote_authentication.txt:** Original Banknote Authentication dataset.
- **README.md:** Project documentation and experimental observations.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run

1. Clone or download the public GitHub repository.
2. Place the Banknote Authentication dataset inside the `data/` directory.
3. Ensure the dataset filename is `data_banknote_authentication.txt`.
4. Install the required Python libraries.
5. Open `notebook.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.
6. Execute the notebook cells sequentially.
7. Review the generated decision boundaries, SVM margins, kernel comparisons, and final performance metrics.

**Note:** The notebook expects the dataset at `data/data_banknote_authentication.txt`, relative to the notebook's working directory.

## Conclusion

This assignment demonstrates the practical implementation and comparison of Decision Trees and Support Vector Machines, covering both linear and nonlinear classification techniques.

The manually implemented Decision Tree achieved 98.18% accuracy, while the manually implemented Linear SVM achieved 86.91%. Among the evaluated models, the Scikit-learn RBF SVM delivered the highest performance with 100% test accuracy.

The project highlights the importance of feature scaling, optimization, decision boundaries, kernel selection, model interpretability, and computational efficiency in machine learning.

---
