# House Price Prediction & Deployment

## Overview

This project builds a complete machine learning pipeline for predicting house prices using the USA Housing dataset. It covers the full workflow from exploratory data analysis (EDA) through model training and comparison, to web deployment with Gradio. The best-performing model achieves a Test R² of **0.9180**, meaning it explains approximately 91.8% of the variance in house prices.

## Dataset

- **Source:** USA Housing dataset
- **Total samples:** 5,000
- **Features used:**
  - Avg. Area Income — Average income of residents in the area
  - Avg. Area House Age — Average age of houses in the area
  - Avg. Area Number of Rooms — Average number of rooms in the area
  - Avg. Area Number of Bedrooms — Average number of bedrooms in the area
  - Area Population — Population of the area
- **Target:** Price (House selling price)
- **Excluded:** Address (text identifier with no predictive value)

## Key Findings from EDA

- **Avg. Area Income** has the strongest correlation with Price (r ≈ 0.64)
- **Avg. Area House Age** has a moderate positive correlation (r ≈ 0.45)
- **Area Population** shows a moderate positive correlation (r ≈ 0.41)
- **Avg. Area Number of Rooms** has a weaker positive correlation (r ≈ 0.34)
- **Avg. Area Number of Bedrooms** has the weakest correlation (r ≈ 0.17)
- All relationships are predominantly **linear** — linear models are expected to perform well
- No missing values or duplicate rows detected
- Price distribution is approximately normal with slight right skew
- No extreme outliers requiring removal

## Model Comparison

| Model | Train R² | Test R² | Test MSE |
|-------|----------|---------|----------|
| Linear Regression | 0.9180 | 0.9180 | $10,089,009,301 |
| Polynomial (degree 2) + Ridge | 0.9181 | 0.9179 | $10,099,222,957 |
| KNN (k=5) | 0.9109 | 0.8693 | $16,078,241,761 |

## Final Model

**Model:** Linear Regression

**Test R²:** 0.9180

**Test MSE:** $10,089,009,301

**Why this model?**

Linear Regression achieved the highest Test R² score of 0.9180, slightly outperforming the Polynomial+Ridge model (0.9179) and significantly outperforming KNN (0.8693). This result is consistent with our EDA findings: the scatter plots showed predominantly linear relationships between features and Price, with no strong non-linear patterns. The near-identical Train and Test R² scores (both 0.9180) confirm the model generalizes well without overfitting. The simplicity and interpretability of Linear Regression also make it an excellent choice — it provides clear coefficient weights that help understand each feature's contribution to the predicted price. KNN's lower Test R² and large Train-Test gap (0.0416) indicate it overfits to the training data and struggles to generalize.

## Web Application

The model is deployed as an interactive web application using Gradio.

### Screenshots

![Gradio Interface](screenshots/model_predictions_comparison.png)

![Model Comparison](screenshots/model_comparison.png)

## Installation

```bash
git clone <your-repo-url>
cd house-price-prediction
pip install -r requirements.txt
```

## Usage

### Run the web app:
```bash
python app.py
```

### Run the notebooks:
```bash
jupyter notebook notebooks/1_eda.ipynb
jupyter notebook notebooks/2_training.ipynb
```

## Project Structure

```
house-price-prediction/
├── data/
│   └── usa_housing.csv
├── notebooks/
│   ├── 1_eda.ipynb
│   └── 2_training.ipynb
├── app.py
├── models/
│   └── best_model.pkl
├── screenshots/
│   ├── price_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── price_correlation_bar.png
│   ├── scatter_plots.png
│   ├── pair_plot.png
│   ├── model_predictions_comparison.png
│   └── model_comparison.png
├── README.md
└── requirements.txt
```

## Technologies Used

- **Python 3.10+**
- **Pandas** & **NumPy** — Data manipulation and analysis
- **Matplotlib** & **Seaborn** — Data visualization
- **Scikit-learn** — Machine learning (Linear Regression, Polynomial Regression, Ridge, KNN, pipelines, metrics)
- **Gradio** — Web application deployment
- **Joblib** — Model serialization
