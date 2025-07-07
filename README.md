# Sensor-based-Bug-Prediction

---

## 📌 Assignment 1 – PCA, P-Values, and Outlier Detection

### ✅ Techniques Applied
- **Principal Component Analysis (PCA)** for dimensionality reduction.
- **P-Value based filtering** to remove statistically insignificant columns.
- **Z-score based outlier detection** to improve model accuracy.

### 🔧 Key Steps
- Loaded multiple CSVs from `consolidated/` folder.
- Removed columns with p-values > 0.05.
- Standardized data and applied PCA (2D & 3D visualization).
- Used z-score thresholding (±3) to drop outliers.

---

## 📌 Assignment 2 – Clustering Techniques

### ✅ Techniques Used

| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| K-Means                | 59.5%    | 0.16      | 0.48   | 0.25     |
| K-Modes                | 74.4%    | 0.23      | 0.36   | 0.28     |
| DBSCAN                 | **82.9%**| 0.21      | 0.09   | 0.13     |
| Gaussian Mixture Model| 53.7%    | 0.20      | 0.76   | 0.31     |

### 🧠 Insights

- **DBSCAN** gave the best accuracy but with low recall.
- **K-Modes** performed well overall, especially on categorical-type data.
- **GMM** showed high recall, suitable for imbalanced bug prediction cases.

### 📂 Dataset
- `con_df.csv`: Software metrics dataset (27221 rows, 8 columns).

---

## 📌 Assignment 3 – Supervised Learning + Bias-Variance Analysis

### ✅ Models Applied

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Decision Tree       | **86%**  | 0.49      | 0.50   | 0.49     |
| Logistic Regression | **86%**  | 0.45      | 0.05   | 0.10     |
| Naive Bayes         | 74%      | 0.26      | 0.41   | 0.32     |

### 📊 Evaluations for Each Model
- **Confusion Matrix**
- **Classification Report**
- **Bias vs Variance Plot**
- **Training vs Testing MSE Plot**

### 📂 Dataset
- `final-data.csv`: Final bug-labeled dataset used for training classifiers.

---

## 📉 Visualizations

- PCA plots (2D & 3D)
- Z-score Outlier Removal Effect
- Confusion Matrices for all classifiers
- Bias vs Variance plots (Decision Tree, Logistic Regression)
- MSE (Train/Test) vs Tree Depth / Regularization Strength

---

## 🛠️ Tools & Libraries

```python
numpy, pandas, matplotlib, seaborn
scikit-learn: PCA, clustering, classification, metrics
kmodes: for K-Modes clustering
