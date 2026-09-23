# Crop Yield Prediction Using Machine Learning

> **IBM SkillsBuild / AICTE Internship Project**

---

## 1. Project Overview

This project builds an end-to-end machine learning pipeline to predict **crop yield in tons per hectare (tons/ha)** from a combination of soil, environmental, and agronomic features. It was completed as part of the IBM SkillsBuild / AICTE internship programme.

---

## 2. Problem Statement

Given a set of agronomic and environmental variables recorded across multiple fields, crops, regions, and seasons, predict the **crop yield (yield_tpha)** in tons per hectare. This is a **supervised regression** problem trained on 4 800 labelled samples and evaluated on 1 200 unseen test samples.

---

## 3. Objectives

- Understand and explore the agricultural dataset
- Clean and preprocess the data appropriately
- Perform Exploratory Data Analysis (EDA) to identify key yield drivers
- Engineer features and encode categorical variables using sklearn pipelines
- Train and compare multiple regression models
- Evaluate models using MAE, RMSE, and R² metrics
- Select the best model based on validation performance
- Generate test predictions in the required Kaggle submission format

---

## 4. Dataset

| Split | Rows | Columns |
|-------|------|---------|
| Training set (`crop_yield_train.csv`) | 4 800 | 18 (includes target) |
| Test set (`crop_yield_test.csv`)       | 1 200 | 17 (no target) |
| Sample submission (`sample_submission.csv`) | 1 200 | 2 |

**No missing values. No duplicate rows.**

---

## 5. Kaggle Dataset Link

[https://www.kaggle.com/competitions/crop-yield-prediction-challenge](https://www.kaggle.com/competitions/crop-yield-prediction-challenge)

---

## 6. Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| soil_ph | float | Soil pH value |
| soil_moisture | float | Soil moisture (%) |
| avg_temperature | float | Average growing-season temperature (°C) |
| total_rainfall | float | Total rainfall (mm) |
| fertilizer_amount | float | Fertilizer applied (kg/ha) — **most important feature** |
| pesticide_usage | float | Pesticide applied (kg/ha) |
| sunlight_hours | float | Total sunlight hours |
| nitrogen_content | float | Soil nitrogen (%) |
| phosphorus_content | float | Soil phosphorus (%) |
| potassium_content | float | Soil potassium (%) |
| irrigation_frequency | int | Irrigations per season (1–6) |
| crop_type | object | Barley / Corn / Rice / Soybean / Wheat |
| region | object | Central / East / North / South / West |
| season | object | Autumn / Spring / Summer |
| harvest_date | object | Harvest date (not used as feature) |
| field_id | object | Field identifier (not used as feature) |
| **yield_tpha** | **float** | **TARGET — Crop yield (tons/ha)** |

**Target statistics:**  
- Min: 2.34 · Mean: 6.27 · Max: 9.54 · Std: 1.15

---

## 7. Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11 | Programming language |
| pandas | 2.2.3 | Data manipulation |
| numpy | 2.4.6 | Numerical computation |
| matplotlib | 3.11.1 | Data visualisation |
| scikit-learn | latest | ML models and pipelines |
| Jupyter Notebook | — | Interactive development |

---

## 8. Project Workflow

```
1. Data Inspection → 2. Data Cleaning → 3. EDA
       ↓
4. Feature Engineering → 5. Train/Val Split (80/20)
       ↓
6. Model Training (LinearRegression, RandomForest, GradientBoosting)
       ↓
7. Model Evaluation (MAE, RMSE, R²) → 8. Model Selection
       ↓
9. Feature Importance → 10. Retrain on Full Data → 11. Test Predictions
```

---

## 9. Exploratory Data Analysis

Key findings from EDA:

- **Fertilizer amount** has the strongest positive correlation with yield
- **Pesticide usage** and **total rainfall** are the next most correlated features
- Yield is approximately normally distributed (mean ≈ 6.27 tons/ha, std ≈ 1.15)
- Yield does not vary dramatically across crop types, regions, or seasons in isolation
- Soil pH, soil moisture, sunlight hours, and temperature show weaker but positive correlations

Visualisations saved in `outputs/`:
- `yield_distribution.png`
- `correlation_heatmap.png`
- `yield_vs_rainfall.png`
- `yield_vs_temperature.png`
- `yield_vs_fertilizer.png`
- `yield_by_crop_type.png`
- `yield_by_region.png`
- `yield_by_season.png`

---

## 10. Machine Learning Models

Three regression models were trained and compared:

1. **Linear Regression** (baseline)
2. **Random Forest Regressor** (n_estimators=200)
3. **Gradient Boosting Regressor** (n_estimators=200) ← **Final model**

All models used a sklearn `Pipeline` with `ColumnTransformer` preprocessing:
- **Numerical features** → `StandardScaler`
- **Categorical features** → `OneHotEncoder(handle_unknown='ignore')`

---

## 11. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error — average absolute prediction error (tons/ha) |
| **RMSE** | Root Mean Squared Error — penalises large errors more heavily |
| **R²** | Coefficient of determination — proportion of variance explained (0–1) |

---

## 12. Actual Results

> All results are from a **20% validation hold-out set** (random_state=42).

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | 0.5236 | 0.6534 | 0.6829 |
| Random Forest | 0.5375 | 0.6705 | 0.6661 |
| **Gradient Boosting** | **0.5212** | **0.6519** | **0.6843** |

**Best Model: Gradient Boosting Regressor**
- MAE = **0.5212 tons/ha**
- RMSE = **0.6519 tons/ha**
- R² = **0.6843**

---

## 13. Key Findings

1. **Fertilizer amount** (importance ≈ 0.768) is by far the most important predictor of crop yield.
2. **Pesticide usage** (≈ 0.085) ranks second.
3. **Total rainfall** (≈ 0.062) ranks third.
4. Soil and environmental features collectively contribute the remaining importance.
5. All three models achieve similar performance (R² 0.67–0.68), suggesting that the dataset has substantial inherent noise that even complex models cannot fully explain.

---

## 14. Project Structure

```
crop-yield-prediction/
│
├── data/
│   ├── crop_yield_train.csv
│   ├── crop_yield_test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── Crop_Yield_Prediction.ipynb
│
├── outputs/
│   ├── crop_yield_predictions.csv
│   ├── yield_distribution.png
│   ├── correlation_heatmap.png
│   ├── yield_vs_rainfall.png
│   ├── yield_vs_temperature.png
│   ├── yield_vs_fertilizer.png
│   ├── yield_by_crop_type.png
│   ├── yield_by_region.png
│   ├── yield_by_season.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   └── residual_analysis.png
│
├── README.md
├── requirements.txt
└── Crop_Yield_Project_Report.docx
```

---

## 15. Installation

```bash
# Clone or navigate to the project directory
cd crop-yield-prediction

# Install required packages
pip install -r requirements.txt
```

---

## 16. How to Run

```bash
# Start Jupyter Notebook
jupyter notebook

# Open:
# notebooks/Crop_Yield_Prediction.ipynb

# Run all cells (Kernel → Restart & Run All)
```

The notebook automatically:
1. Loads the data from `data/`
2. Performs EDA and saves plots to `outputs/`
3. Trains and compares all models
4. Saves predictions to `outputs/crop_yield_predictions.csv`

---

## 17. Output

| File | Description |
|------|-------------|
| `outputs/crop_yield_predictions.csv` | Final predictions for 1 200 test samples |
| `outputs/model_results.csv` | Model evaluation metrics |
| `outputs/feature_importance.csv` | Feature importance scores |
| `outputs/*.png` | 12 visualisation plots |

---

## 18. Limitations

- R² ≈ 0.68 means ~32% of yield variance is unexplained by available features
- Data covers a single harvest year (2021) — may not generalise to all years
- No hyperparameter tuning was performed
- Region labels are coarse (5 broad regions vs. precise GPS coordinates)
- Identifier columns (`harvest_date`, `field_id`) were dropped and may encode useful signals

---

## 19. Future Scope

- Train XGBoost / LightGBM / CatBoost for potentially better performance
- Systematic hyperparameter tuning with GridSearchCV or Optuna
- SHAP-based explainability for individual predictions
- Streamlit web application for real-time farmer use
- Integration with OpenWeatherMap / NASA POWER weather APIs
- Satellite imagery (NDVI) as additional feature input

---

## 20. Conclusion

The **Gradient Boosting Regressor** was selected as the final model with R² = 0.6843, RMSE = 0.6519 tons/ha, and MAE = 0.5212 tons/ha on the validation set. The model identifies **fertilizer amount** as the dominant predictor of crop yield, followed by **pesticide usage** and **total rainfall**. The project demonstrates a complete end-to-end ML workflow from raw CSV data to a Kaggle-format submission file.

---

## 21. Author

**Suraj**  
IBM SkillsBuild / AICTE Internship  
Project: Crop Yield Prediction Using Machine Learning  
Dataset: [Kaggle Crop Yield Prediction Challenge](https://www.kaggle.com/competitions/crop-yield-prediction-challenge)
