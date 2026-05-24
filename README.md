# Bank Customer Churn — End-to-End Analysis

A complete data-science project on the
[Bank Customer Churn dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
from Kaggle.

The project is split into three self-contained notebooks so each step can be
inspected and re-run independently:

| # | Notebook | What it does |
|---|----------|--------------|
| 01 | `notebooks/01_EDA.ipynb`            | Deep exploratory data analysis — univariate, bivariate, multivariate, statistical tests, customer segmentation. |
| 02 | `notebooks/02_Prediction.ipynb`     | Regression task — predicting `EstimatedSalary` benchmarked across 7 algorithms with cross-validated tuning. |
| 03 | `notebooks/03_Classification.ipynb` | Churn classification — feature engineering, SMOTE, 6-model benchmark, hyper-parameter search, threshold tuning, SHAP explainability. |

## Project structure

```
bank_churn_project/
│── 01_EDA.ipynb
│── 02_Prediction.ipynb
│── 03_Classification.ipynb
|──README.md
|──LICENSE.md
```

## Quick start

### Option A — Google Colab (recommended)

1. Open the notebook on Colab (`File → Open notebook → GitHub` and paste this repo URL).
2. In the first cell of each notebook, upload `Churn_Modelling.csv` or mount Google Drive.
3. `Runtime → Run all`.

### Option B — Local

```bash
git clone <your-repo-url>
cd bank_churn_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# place the dataset
mv /path/to/Churn_Modelling.csv data/

jupyter lab
```

## Dataset

`Churn_Modelling.csv` — 10 000 bank customers with the following columns:

| Column | Description |
|--------|-------------|
| CreditScore | Credit score |
| Geography | Country (France / Spain / Germany) |
| Gender | Male / Female |
| Age | Age in years |
| Tenure | Years as a customer |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Has a credit card (0/1) |
| IsActiveMember | Active member (0/1) |
| EstimatedSalary | Estimated annual salary |
| **Exited** | Target — 1 if the customer churned |

## Key results

* **Classification** — XGBoost / LightGBM reach a 5-fold cross-validated
  ROC-AUC ≈ 0.87 with engineered features. Threshold tuning lifts F1
  (class 1) from ~0.55 to ~0.63.
* **Regression** — `EstimatedSalary` is essentially noise w.r.t. the other
  features (R² ≈ 0). This finding itself is useful: the variable can be
  dropped without harming downstream models.
* **EDA insights** — strongest churn drivers are `Age` (45–60 bracket),
  `Geography = Germany`, `NumOfProducts ≥ 3`, `IsActiveMember = 0` and
  `Gender = Female`.

## Reproducibility

* `random_state = 42` everywhere.
* All pre-processing happens inside `Pipeline` / `ColumnTransformer`
  objects, so no leakage between train and test folds.
* SMOTE oversampling lives inside an `imblearn.pipeline.Pipeline`,
  guaranteeing that it is only applied to training folds during CV.

## License

MIT — see `LICENSE`
