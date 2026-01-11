# California Housing Price Prediction GoldenKeys

**Author:** Ani Kelenjeridze
**Course:** Data Science with Python

## Project Overview

This project analyzes and predicts median house values in California using the built-in California Housing dataset from scikit-learn. The goal is to build machine learning models that can accurately predict house prices based on various features like median income, location, house age, and other demographic factors.

## Problem Statement

Housing prices in California vary significantly based on location, demographics, and property characteristics. This project aims to:

1. Understand which factors most influence house prices
2. Build predictive models to estimate house values
3. Compare different machine learning approaches for price prediction

## Dataset Description

**Source:** Built-in California Housing dataset (sklearn.datasets)

**Size:** 20,640 samples, 8 features + 1 target variable

**Features:**

- `MedInc`: Median income in block group (in tens of thousands)
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

**Target Variable:**

- `MedHouseVal`: Median house value (in hundreds of thousands of dollars)

**Note:** House values are capped at $500,000 in this dataset.

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. Clone this repository:

```bash
https://github.com/Anikelenjeridze/GoldenKeysFinal.git
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

### Required Packages

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
```

## Project Structure

```
california-housing-prediction/
│
├── data/
│   ├── raw/                          # Original dataset
│   │   └── california_housing_raw.csv
│   └── processed/                    # Cleaned dataset
│       └── california_housing_clean.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Initial data exploration
│   ├── 02_data_preprocessing.ipynb   # Data cleaning
│   ├── 03_eda_visualization.ipynb    # Exploratory analysis
│   └── 04_machine_learning.ipynb     # ML models
│
├── src/
│   ├── data_processing.py            # Data cleaning functions
│   ├── visualization.py              # Plotting functions
│   └── models.py                     # ML model implementations
│
├── reports/
│   ├── figures/                      # Generated visualizations
│   └── data_quality_report.txt       # Data quality summary
│
├── README.md                         # This file
├── CONTRIBUTIONS.md                  # Contribution details
└── requirements.txt                  # Python dependencies
```

## Usage

### Running the Analysis

1. **Data Exploration:**

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

2. **Data Preprocessing:**

```bash
jupyter notebook notebooks/02_data_preprocessing.ipynb
```

3. **Exploratory Data Analysis:**

```bash
jupyter notebook notebooks/03_eda_visualization.ipynb
```

4. **Machine Learning Models:**

```bash
jupyter notebook notebooks/04_machine_learning.ipynb
```

### Quick Start Example

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression





# Load data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Prepare features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

## Results Summary

### Key Findings

1. **Strongest Predictors:**

   - Median income (correlation: ~0.69)
   - Geographic location (latitude/longitude)
   - Average number of rooms

2. **Model Performance:**

| Model             | R² Score | RMSE  | MAE   |
| ----------------- | -------- | ----- | ----- |
| Linear Regression | 0.606    | 0.734 | 0.533 |
| Decision Tree     | 0.615    | 0.726 | 0.512 |

3. **Insights:**
   - Income is the most important factor in determining house prices
   - Coastal areas command higher prices
   - Both models achieve reasonable prediction accuracy (R² > 0.60)
   - Decision Tree slightly outperforms Linear Regression
   - Typical prediction error: ~$73,000

### Visualizations

The project includes 9 comprehensive visualizations:

1. Distribution of house prices
2. Correlation heatmap
3. Income vs price relationship
4. Geographic distribution map
5. Feature distributions
6. Price category analysis
7. House age vs price
8. Prediction comparison
9. Residual analysis

## Methodology

### 1. Data Processing

- Checked for missing values (none found)
- Analyzed outliers using IQR method
- Created derived features (rooms per household, etc.)
- Handled any infinite values from calculations

### 2. Exploratory Data Analysis

- Statistical summaries using pandas `.describe()`
- Correlation analysis with heatmaps
- Distribution plots for all features
- Geographic visualization of price patterns
- Relationship analysis between features and target

### 3. Machine Learning

- **Models Used:** Linear Regression, Decision Tree Regressor
- **Train-Test Split:** 80% training, 20% testing
- **Evaluation Metrics:** R², MSE, RMSE, MAE
- **Feature Selection:** Used 8 original features
- **Model Comparison:** Both models evaluated and compared

### 4. Evaluation

- Comprehensive performance metrics
- Residual analysis
- Feature importance analysis
- Prediction vs actual value plots

## Limitations

1. House values are capped at $500,000 (data collection limitation)
2. Dataset is from 1990 census data (not current prices)
3. Block-level aggregation may hide individual property variations
4. No information on property condition or amenities
5. Limited to California only

## Future Work

Potential improvements for this project:

- Try additional models (Random Forest, Gradient Boosting)
- Implement hyperparameter tuning
- Add cross-validation for more robust evaluation
- Create an interactive dashboard for predictions
- Incorporate more recent housing data
- Add feature scaling/normalization
- Explore polynomial features for non-linear relationships

## Technologies Used

- **Python 3.11**
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations
- **Matplotlib/Seaborn:** Data visualization
- **Scikit-learn:** Machine learning models and evaluation
- **Jupyter:** Interactive notebook environment
