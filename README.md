# 🌲 Forest Cover Type Prediction

A machine learning project that predicts forest cover type from cartographic variables using data from four wilderness areas in the Roosevelt National Forest, northern Colorado.

---

## 📌 Overview

This project tackles a **multiclass classification problem**: given terrain and soil features derived from cartographic data, predict which of 7 forest cover types occupies a 30×30 meter cell. No remotely sensed data is used — only variables from USGS and USFS cartographic records.

The study areas (Rawah, Neota, Comanche Peak, and Cache la Poudre) represent minimally human-disturbed ecosystems, meaning cover types reflect natural ecological processes rather than forest management.

---

## 📂 Dataset

- **Source:** US Forest Service (USFS) Region 2 Resource Information System (RIS) + USGS cartographic data
- **Observations:** ~581,012 records (30×30 m cells)
- **Features:** 54 columns — continuous terrain variables + binary one-hot encoded soil types (40) and wilderness areas (4)
- **Target:** `Cover_Type` — 7 classes:

| Type | Species |
|------|---------|
| 1 | Spruce/Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood/Willow |
| 5 | Aspen |
| 6 | Douglas-Fir |
| 7 | Krummholz |

> The dataset is **imbalanced** — types 3–7 are underrepresented compared to types 1 and 2.

---

## 🗂️ Project Structure

```
forest-cover-type/
│
├── forest.ipynb          # Main analysis notebook
├── covertype.csv         # Dataset
└── README.md
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Inspected class distribution of `Cover_Type`
- Checked for null values and duplicates (none found)
- Computed correlation heatmap (excluding binary soil type columns) to identify redundant features

### 2. Feature Engineering
- **Scaling:** tested MinMax normalization and Standard scaling
- **Feature selection:** dropped low-correlation distance features (`Horizontal_Distance_To_Hydrology`, `Vertical_Distance_To_Hydrology`, `Horizontal_Distance_To_Roadways`, `Horizontal_Distance_To_Fire_Points`) and unnamed columns
- **Key finding:** removing features consistently hurt performance — the full unscaled dataset produced the best results across all models

### 3. Sampling Strategy
Due to the large dataset size (training times exceeding 10 minutes), a **25% stratified sample** (`random_state=23`) was used for model comparison and hyperparameter tuning, with final evaluation on the full dataset.

### 4. Models Evaluated

| Model | Best Accuracy (sample) |
|-------|----------------------|
| KNN (k=5) | 0.92 |
| Decision Tree (max_depth=20) | 0.87 |
| Random Forest (100 trees, max_depth=20) | 0.87 |
| AdaBoost + Decision Tree | 0.93 |
| XGBoost | 0.94 |

### 5. Hyperparameter Tuning
Tuning was performed using `RandomizedSearchCV` (cv=2, n_iter=5) on KNN, AdaBoost, and XGBoost.

**KNN tuning results (varying k):**

| k | Accuracy (no scaling) |
|---|----------------------|
| 1 | 0.94 |
| 3 | ~0.93 |
| 5 | 0.92 |
| 10 | ~0.91 |
| 20 | ~0.89 |

KNN with `k=1` on the raw full dataset outperformed tuned AdaBoost and XGBoost while being significantly cheaper to train.

---

## ✅ Final Model

**K-Nearest Neighbors with k=1**, trained on the complete unscaled dataset.

```python
KNeighborsClassifier(n_neighbors=1)
```

- **Test accuracy:** 96.90%
- **Train accuracy:** 100% *(expected — with k=1, the model's nearest neighbor during training is the point itself)*

### On Overfitting
The 100% training score is not a concern here: with k=1, each training point is its own nearest neighbor, so the training score is structurally 1.0. The 96.9% test accuracy — on unseen data — is the meaningful metric.

### On Class Imbalance
The confusion matrix showed that the model correctly identifies minority cover types (4 and 5) despite their low representation in the dataset, which may partly be a consequence of the k=1 memorisation behaviour.

---

## 📊 Key Findings

- **Raw data outperforms engineered data** — scaling and feature removal consistently reduced accuracy across all models tested
- **KNN dominates** — despite its simplicity, KNN with k=1 achieved the highest accuracy (96.9%) at a fraction of the computational cost of ensemble methods
- **Ensemble methods underperformed** — AdaBoost peaked at 93% and XGBoost at 94% on the sample; neither matched KNN on the full dataset after tuning

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas`, `numpy` — data manipulation
  - `matplotlib`, `seaborn` — visualisation
  - `scikit-learn` — ML models, preprocessing, evaluation
  - `xgboost` — gradient boosting

---

## ▶️ How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

2. Place `covertype.csv` in the project root.

3. Open and run `forest.ipynb` in Jupyter or any compatible environment.

> ⚠️ Running the KNN model on the full dataset takes approximately 10 minutes. Use the 25% sample (`df.sample(frac=0.25, random_state=23)`) for faster experimentation.

---

## 📊 Link to the slides

[Googles Slides](https://docs.google.com/presentation/d/1205rHM5hQ96E4N8c5VLgZePFl1BShwRYR3Wbl9Nuzjk/edit?usp=sharing)
---

## 👥 Authors

Jorge Barrios Hurtado