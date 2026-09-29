# 🏢 Smart Building Energy Efficiency & Carbon Footprint Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Green?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-blueviolet?style=for-the-badge)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade Machine Learning solution designed to predict building energy consumption (**Heating & Cooling Loads**) and calculate environmental impact (**Carbon Footprint in $\text{kg CO}_2$**). By combining architectural parameters from the **UCI Energy Efficiency Dataset** with real-time weather enrichment via the **Visual Crossing Weather API**, this repository provides actionable intelligence for smart building energy optimization, ESG compliance, and green building certification (LEED/BREEAM).

---

## 📌 Table of Contents
- [Executive Summary](#-executive-summary)
- [Key Features](#-key-features)
- [Dataset & Feature Engineering](#-dataset--feature-engineering)
- [Machine Learning Architecture & Model Zoo](#-machine-learning-architecture--model-zoo)
- [Model Performance & Diagnostics](#-model-performance--diagnostics)
- [Model Explainability (XAI)](#-model-explainability-xai)
- [Getting Started](#-getting-started)
- [Business Impact & Real-World Application](#-business-impact--real-world-application)
- [License & Contact](#-license--contact)

---

## 🎯 Executive Summary

Buildings account for approximately **30% of global primary energy consumption** and **39% of energy-related carbon emissions**. Optimizing HVAC systems and architectural design prior to construction is critical for net-zero carbon goals.

This project delivers:
1. **Multi-Target Energy Modeling**: Accurately forecasts total building energy usage ($\text{kWh}$) based on 8 core structural parameters.
2. **Dynamic Weather API Enrichment**: Merges localized micro-climate data (temperature, humidity, solar radiation, HDD/CDD) into building physics profiles.
3. **Comprehensive Benchmark Zoo**: Evaluates 9 regression algorithms ranging from baseline linear models to ensemble tree methods (`LightGBM`, `XGBoost`, `Random Forest`) and a meta-learned `StackingRegressor`.
4. **Hyperparameter Tuning & Explainability**: Implements `RandomizedSearchCV` cross-validation alongside SHAP (SHapley Additive exPlanations) for transparent model auditing.
5. **Carbon Emission Accounting**: Converts predicted energy consumption directly into carbon metrics ($\text{kg CO}_2$) using standard emissions intensity factors ($0.233\text{ kg CO}_2\text{/kWh}$).

---

## ✨ Key Features

- **Automated Ingestion & Cleaning**: Downloads UCI dataset programmatically, handling schema normalization, type coercion, and missing value checks.
- **Weather API Integration**: Fetches daily timeline data from Visual Crossing API and engineers **Heating Degree Days ($\text{HDD}$)** and **Cooling Degree Days ($\text{CDD}$)** (base temp $18^\circ\text{C}$).
- **Rigorous EDA Suite**: Automated generation of feature distributions, correlation heatmaps, and target interaction plots.
- **Production Preprocessing**: Scikit-Learn `ColumnTransformer` pipeline with `StandardScaler` for numeric features and `OneHotEncoder` for categorical variables to prevent data leakage.
- **Advanced Model Zoo**:
  - Linear, Ridge, Lasso Regression
  - Random Forest & Gradient Boosting Regressors
  - XGBoost & LightGBM Regressors
  - Multi-Layer Perceptron (MLP) Neural Network
  - Custom Stacking Regressor (RF + LightGBM meta-fitted with Ridge)
- **Tuning & Diagnostics**: Automated hyperparameter optimization for `LightGBM` using 5-fold cross-validation with RMSE loss optimization. Generates Residuals vs. Fitted and Predicted vs. Actual diagnostic plots.
- **Explainable AI (XAI)**: SHAP TreeExplainer beeswarm plots, feature contribution bars, and Permutation Feature Importance.
- **Artifact Pipeline**: Saves metrics CSVs, diagnostic plots, trained model pipelines, and predictions reports.

---

## 📊 Dataset & Feature Engineering

### 1. UCI Energy Efficiency Dataset (`ENB2012_data.xlsx`)
Simulated building shape parameters generated using Ecotect software (Tsanas & Xifara, 2012):

| Variable | Feature Name | Description | Range / Values |
| :--- | :--- | :--- | :--- |
| **$X_1$** | `Relative_Compactness` | Ratio of building volume to surface area | $0.62 - 0.98$ |
| **$X_2$** | `Surface_Area` | Total building envelope surface area ($\text{m}^2$) | $514.5 - 808.5\text{ m}^2$ |
| **$X_3$** | `Wall_Area` | Total exterior wall area ($\text{m}^2$) | $245.0 - 416.5\text{ m}^2$ |
| **$X_4$** | `Roof_Area` | Total roof surface area ($\text{m}^2$) | $110.25 - 220.5\text{ m}^2$ |
| **$X_5$** | `Overall_Height` | Total building height ($\text{m}$) | $3.5\text{ m}$ (1-story) / $7.0\text{ m}$ (2-story) |
| **$X_6$** | `Orientation` | Cardinal orientation facing direction | $2$: N, $3$: E, $4$: S, $5$: W |
| **$X_7$** | `Glazing_Area` | Ratio of window area to floor area | $0.00, 0.10, 0.25, 0.40$ |
| **$X_8$** | `Glazing_Area_Distribution` | Window arrangement pattern | $0$: None, $1$: Uniform, $2\dots 5$: Orientations |
| **$Y_1$** | `Heating_Load` | Energy required to heat building ($\text{kWh/m}^2$) | $6.01 - 43.10\text{ kWh/m}^2$ |
| **$Y_2$** | `Cooling_Load` | Energy required to cool building ($\text{kWh/m}^2$) | $10.90 - 48.03\text{ kWh/m}^2$ |

### 2. Engineered Target & Carbon Metrics
- **Total Energy Usage**: $\text{Energy Usage (kWh)} = \text{Heating Load} + \text{Cooling Load}$
- **Carbon Footprint**: $\text{Carbon Emissions (kg CO}_2\text{)} = \text{Energy Usage (kWh)} \times 0.233\text{ kg CO}_2\text{/kWh}$

### 3. Weather API Enrichment Features
When `USE_WEATHER = True`, real-time daily weather stats are joined via Cartesian merge across target cities:
- `temp` (Average Temperature), `humidity`, `dew`, `precip`, `pressure`, `cloudcover`, `solarradiation`, `windspeed`.
- **Heating Degree Days ($\text{HDD}$)**: $\max(0, 18.0 - \text{temp})$
- **Cooling Degree Days ($\text{CDD}$)**: $\max(0, \text{temp} - 18.0)$

---

## 🏗 System Architecture & Pipeline

```
┌────────────────────────────────┐     ┌────────────────────────────────┐
│   UCI Dataset (ENB2012)        │     │  Visual Crossing Weather API   │
│   (Building Physics Features)  │     │  (Temp, Humidity, HDD/CDD)     │
└──────────────┬─────────────────┘     └──────────────┬─────────────────┘
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
                     ┌────────────────────────┐
                     │ Preprocessing Pipeline │
                     │  - StandardScaler      │
                     │  - OneHotEncoder       │
                     └────────────┬───────────┘
                                  ▼
                     ┌────────────────────────┐
                     │   Train/Test Split     │
                     │   (80% Train / 20% Test│
                     └────────────┬───────────┘
                                  ▼
                     ┌────────────────────────┐
                     │     Model Zoo &        │
                     │ Hyperparameter Tuning  │
                     │ (RandomizedSearchCV)   │
                     └────────────┬───────────┘
                                  ▼
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│ Performance      │                            │ Explainability   │
│ Evaluation       │                            │ & Impact         │
│ (RMSE, MAE, R²)  │                            │ (SHAP, CO₂ Est.) │
└──────────────────┘                            └──────────────────┘
```

---

## ⚙ Machine Learning Architecture & Model Zoo

The project benchmarks multiple regression paradigms encapsulated inside scikit-learn `Pipeline` objects:

1. **Linear Regressors**: `LinearRegression`, `Ridge(alpha=1.0)`, `Lasso(alpha=0.01)`
2. **Ensemble Trees**: `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBRegressor`, `LGBMRegressor`
3. **Neural Network**: `MLPRegressor(hidden_layer_sizes=(64, 32))`
4. **Stacking Ensemble**: Combined `RandomForest` and `LightGBM` base estimators with a `Ridge` meta-regressor.

### Hyperparameter Tuning (`LightGBM`)
Optimal hyperparameters for `LightGBM` are discovered using `RandomizedSearchCV` over 50 iterations with 5-fold cross-validation:
- `est__n_estimators`: `[200, 400, 600, 800, 1000]`
- `est__num_leaves`: `[15, 31, 63, 127]`
- `est__learning_rate`: `[0.01, 0.03, 0.05, 0.1]`
- `est__subsample`: `[0.6, 0.7, 0.8, 0.9, 1.0]`
- `est__colsample_bytree`: `[0.6, 0.7, 0.8, 0.9, 1.0]`

---

## 📈 Model Performance & Diagnostics

Evaluation is conducted on the held-out test set ($20\%$) using standard metrics:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}, \quad \text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|, \quad R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

Diagnostic plots generated automatically during execution:
- **Predicted vs. Actual Plot**: Measures goodness-of-fit along the identity line ($y = x$).
- **Residuals vs. Fitted Plot**: Checks for homoscedasticity and non-linear residual patterns.

*All metrics are automatically exported to `artifacts/metrics_energy.csv`.*

---

## 🔍 Model Explainability (XAI)

To ensure model transparency for architectural engineers and sustainability consultants, the framework integrates **SHAP**:

- **SHAP Beeswarm Plot** (`figures/07_shap_summary.png`): Illustrates feature impact direction and magnitude across individual building predictions.
- **SHAP Feature Importance Bar Chart** (`figures/08_shap_bar.png`): Ranks overall feature importance globally.
- **Permutation Importance** (`figures/09_perm_importance.png`): Fallback importance metric measuring drop in score when individual features are shuffled.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / JupyterLab / Google Colab

### Installation & Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/smart-building-energy-prediction.git
   cd smart-building-energy-prediction
   ```

2. **Install Required Packages**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm shap requests openpyxl
   ```

3. **Configure API Key (Optional for Weather Enrichment)**:
   - Obtain a free API key from [Visual Crossing Weather API](https://www.visualcrossing.com/weather-api).
   - Set the key in your environment or Colab secrets:
     ```python
     os.environ["VC_API_KEY"] = "YOUR_API_KEY_HERE"
     ```
   - *Note: If no API key is set, the notebook seamlessly runs in standalone building mode.*

4. **Run the Execution Notebook**:
   ```bash
   jupyter notebook Energy-Efficiency-Carbon-Emission-Prediction.ipynb
   ```

---

## 💡 Business Impact & Real-World Application

1. **Sustainable Architecture & Retrofitting**: Enables real estate developers to simulate energy loads during pre-construction design, optimizing `Glazing_Area` and `Overall_Height` to minimize HVAC demand.
2. **ESG & Carbon Reporting**: Automates $\text{CO}_2$ emissions reporting for corporate sustainability disclosures (Scope 1 & Scope 2 emissions).
3. **Smart Grid & Load Forecasting**: Offers utility companies insights into building-level energy demands under changing climate conditions.

---

## 📜 License & Acknowledgments

- **Dataset**: UCI Machine Learning Repository — Energy Efficiency Dataset (Tsanas & Xifara, 2012).
- **Weather Data**: Visual Crossing Weather API.
- **License**: Released under the [MIT License](LICENSE).

---

<p align="center">
  <i>Developed with ❤️ for Sustainable Engineering & Applied Machine Learning.</i>
</p>


