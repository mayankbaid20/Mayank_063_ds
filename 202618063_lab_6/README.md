# DS605 Lab Assignment 6

## Feature Extraction and Machine Learning with Image and Text Data

This project implements traditional machine learning techniques for **image and text classification** using extracted numerical features.

### 📁 Project Structure

```text
202618063_lab_6/
│
├── 448/
│   └── Asphalt crack images
│
├── emails.csv
├── work.ipynb
└── README.md
```

### 🖼️ Part A — Image Classification

The asphalt crack images are processed using **OpenCV**.

Features extracted:

* Mean brightness
* Contrast
* Dark pixel ratio
* Bright pixel ratio
* Canny edge count
* Canny edge density

Images are resized to `128 × 128` and converted to grayscale.

A **Random Forest Classifier** is used for classification.

Evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Training and prediction time

### 📧 Part B — Email Classification

The email dataset contains pre-existing **word-count features** and a `Prediction` column representing the class.

A **Logistic Regression** classifier is used with:

1. Existing word-count features
2. TF-IDF transformed features

Both representations are evaluated using accuracy, precision, recall, F1-score and computation time.

### 🔧 Part C — Improved Representation

An improved TF-IDF representation using **sublinear term frequency** is implemented and compared with the original word-count and TF-IDF representations.

### 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Scikit-learn
* Jupyter Notebook

### ▶️ How to Run

1. Clone/download the repository.
2. Keep `448/`, `emails.csv`, and `work.ipynb` in the same directory.
3. Install the required libraries:

```bash
pip install numpy pandas opencv-python matplotlib scikit-learn
```

4. Open `work.ipynb` in Jupyter Notebook or VS Code.
5. Run all cells sequentially.

### 📌 Note

The provided `emails.csv` is already represented as numerical word-count features, so TF-IDF is applied using `TfidfTransformer` rather than `CountVectorizer`.
