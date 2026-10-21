# DS605: Fundamentals of Machine Learning
## Lab Assignment 7 – Image Representation, PCA, Eigenfaces and Face Recognition

**Name:** Mayank Baid  
**Dataset:** AT&T Database of Faces (ORL)  
**Repository:** `Mayank_063_ds`

## Project Overview

This project implements image representation, dimensionality reduction using Principal Component Analysis (PCA), eigenface visualization, face reconstruction, and face recognition using the AT&T Database of Faces (ORL).

The primary objective is to represent facial images as numerical feature vectors, reduce their dimensionality while preserving important information, and evaluate face recognition performance using similarity-based matching and traditional machine-learning classifiers.

PCA is implemented using NumPy linear-algebra operations, including eigendecomposition and Singular Value Decomposition (SVD). Scikit-learn is used only for the final traditional machine-learning comparison.

## Pipeline Architecture & Workflow

### 1. Part A: Image Representation

- **Dataset Loading:** Loaded facial images from the AT&T Database of Faces (ORL), organized into subject-specific folders.
- **Grayscale Conversion:** Converted images into grayscale intensity matrices using Pillow.
- **Feature Extraction:** Flattened each image into a numerical feature vector representing its pixel intensities.
- **Dataset Statistics:**
  - Total images: 400
  - Image dimensions: 112 × 92 pixels
  - Features per image: 10,304
  - Observed pixel intensity range: 0–251
- **Subject Labels:** Assigned numerical labels to identify the subjects associated with each image.
- **Train-Test Split:** Implemented a reproducible, stratified split using random seed 42, ensuring that every subject was represented in both sets.
  - Training samples: 280 (70%)
  - Testing samples: 120 (30%)

### 2. Part B: PCA, Eigenfaces and Compression

- **Mean Face Calculation:** Calculated the mean face from the training data and centered the training and test images using the training mean.
- **Eigendecomposition:** Applied the dual-space covariance approach to compute principal directions efficiently, avoiding direct decomposition of the much larger pixel-space covariance matrix.
- **SVD Comparison:** Computed principal directions using NumPy's SVD and compared the resulting eigenvalues with those obtained through eigendecomposition. The top five eigenvalues matched the corresponding squared singular values divided by the degrees of freedom.
- **Explained Variance Analysis:** Plotted cumulative explained variance and identified the minimum number of principal components required to preserve at least 90% of the variance.
- **Component Selection:** Retained 92 principal components to achieve the 90% explained-variance threshold.
- **Eigenface Visualization:** Reshaped the principal directions into image dimensions to visualize the top 10 eigenfaces.
- **PCA Projection:** Projected the training and test images into the reduced PCA feature space.
- **Face Reconstruction:** Reconstructed sample images from their reduced representations and compared them visually with the original images.
- **Storage Comparison:** Compared the storage requirements of the original training pixel matrix with the PCA representation, including the retained eigenfaces and mean face.

**Results:**
- Original feature dimensions: 10,304
- Reduced feature dimensions: 92
- Storage savings: 65.89%, based on the notebook's calculated storage comparison.

### 3. Part C: Face Recognition in PCA Space

- **Similarity-Based Matching:** Implemented a nearest-neighbor face-matching approach using Euclidean distance.
- **PCA Recognition:** Projected test images into PCA space and matched each image against the stored training representations.
- **Raw-Pixel Recognition:** Repeated nearest-neighbor matching using the original, unreduced pixel vectors.
- **Accuracy Evaluation:** Compared recognition accuracy for both representations on the same test set.

**Recognition Results:**

| Approach | Feature Dimensions | Accuracy |
|---|---:|---:|
| 1-NN on raw pixels | 10,304 | 98.33% |
| 1-NN on PCA features | 92 | 98.33% |

The results show that PCA reduced the feature dimensionality substantially while maintaining the same observed recognition accuracy as the raw-pixel baseline.

### 4. Part D: Traditional Machine-Learning Comparison

Trained and evaluated traditional classifiers using the PCA-reduced feature representations.

The following models were compared:

- **Baseline 1-NN:** Euclidean-distance similarity matching in PCA space.
- **K-Nearest Neighbors (KNN):** Used 3 neighbors.
- **Logistic Regression:** Trained with a maximum of 1,000 iterations.
- **Support Vector Machine (SVM):** Used an RBF kernel with C = 10.

**Final Model Comparison:**

| Model | Accuracy |
|---|---:|
| Baseline 1-NN (PCA Euclidean) | 98.33% |
| KNN (k = 3) | 92.50% |
| Logistic Regression | 98.33% |
| Support Vector Machine (RBF) | 100.00% |

The RBF-kernel SVM achieved the highest observed accuracy of 100% on the test set, followed by the nearest-neighbor baseline and Logistic Regression at 98.33%.

These results reflect the current train-test split and model settings; they do not guarantee the same performance on other splits or unseen datasets.

## Key Insights and Observations

- **Effective Dimensionality Reduction:** PCA reduced each facial image from 10,304 pixel features to 92 principal components while retaining at least 90% of the training-data variance.
- **Comparable Recognition Performance:** Raw-pixel and PCA-based nearest-neighbor matching both achieved 98.33% accuracy, demonstrating that the reduced representation retained useful facial information for recognition.
- **Storage Efficiency:** The notebook reported 65.89% storage savings when accounting for the reduced training representations, retained eigenfaces, and mean face.
- **Eigenface Interpretation:** The principal components capture major patterns of variation in facial appearance. Eigenfaces represent directions of variation rather than individual people's faces.
- **Reconstruction Quality:** Reconstructed images retain recognizable facial structure but appear smoother and less detailed than the originals because only a subset of principal components is retained.
- **Classifier Performance:** The RBF-kernel SVM achieved 100% test accuracy, while Logistic Regression matched the PCA nearest-neighbor baseline. KNN with three neighbors performed less accurately on this split.
- **Methodological Comparison:** The project demonstrates how numerical linear algebra can be used to implement PCA independently of Scikit-learn's PCA implementation and how the resulting features can support downstream classification.

## Important Implementation Details

- PCA and eigenface calculations use NumPy linear-algebra operations rather than `sklearn.decomposition.PCA`.
- Both eigendecomposition and SVD are implemented and compared.
- The training-set mean is used to center both training and test images.
- The train-test split is stratified and reproducible, with a 70:30 ratio and random seed 42.
- Scikit-learn is used only for the traditional machine-learning classifier comparison.
- Recognition accuracy is evaluated on the held-out test set.

## Project Structure

```text
Mayank_063_ds/
├── notebook.ipynb
├── eigenfaces.png
├── explained_variance.png
├── reconstruction_comparison.png
├── README.md
└── data/
    ├── s1/
    ├── s2/
    ├── ...
    └── s40/
```

The `data/` directory should contain the ORL dataset's subject folders and image files. The three PNG figures are generated by the notebook and should be included in the public repository.

## Requirements

- Python 3
- NumPy
- Matplotlib
- Pillow
- Scikit-learn

Install the required libraries using:

```bash
pip install numpy matplotlib pillow scikit-learn
```

## How to Run

1. Clone or download the public GitHub repository.
2. Place the ORL dataset in the `data/` directory, maintaining the subject-folder structure expected by the notebook.
3. Install the required Python libraries.
4. Open `notebook.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
5. Run the notebook cells sequentially to load the dataset, perform image representation, calculate PCA and eigenfaces, reconstruct faces, evaluate recognition accuracy, and compare classifiers.
6. Confirm that the output figures are generated and that the reported results are visible in the notebook.

**Note:** The notebook uses `DATA_DIR = './data'`, so the dataset directory must be accessible relative to the notebook's working directory.

## Conclusion

This assignment demonstrates a complete face-recognition workflow, from grayscale image representation and feature extraction to PCA-based dimensionality reduction, eigenface visualization, image reconstruction, similarity-based recognition, and traditional machine-learning classification.

The experiments show that reducing the image representation to 92 principal components preserved the observed nearest-neighbor recognition accuracy of 98.33%, while the RBF-kernel SVM achieved 100% accuracy on the held-out test set. The project highlights the practical benefits of PCA for reducing feature dimensionality while retaining information useful for face recognition.

---

**Course:** DS605 – Fundamentals of Machine Learning  
**Assignment:** Lab Assignment 7 – Image Representation, PCA, Eigenfaces and Face Recognition  
**Submission:** Public GitHub repository link only
