# CreditWise Loan System

A machine learning system that predicts whether a loan application should be **Approved** or **Rejected**, built for a fictional bank (SecureTrust Bank) that wants to replace slow, inconsistent manual loan verification with fast, data-driven decisions.

## Problem
Manual checking of income proofs, credit history and employment details is slow and inconsistent. As a result, good customers get rejected and high-risk customers sometimes get approved. This project learns patterns from historical applications to support the loan officer's decision.

## Dataset
- 1,000 loan applicants, 20 columns (income, credit score, DTI ratio, savings, collateral, loan amount/term/purpose, property area, education, etc.)
- Target column: `Loan_Approved` (Yes / No)
- **The dataset is not included in this repository.** Place `loan_approval_data.csv` in the same folder as the notebook before running it.

## Approach (see `Credit_wise.ipynb`)
1. **Handle missing values**: mean for numeric columns, most-frequent for categorical columns
2. **EDA**: class balance, income/credit-score distributions, box plots for outliers
3. **Encoding**: label encoding for `Education_Level` and the target; one-hot encoding for the other categorical columns
4. **Correlation heatmap** to see which features relate most to approval
5. **Train/test split** (80/20, `random_state=42`) and **StandardScaler**
6. **Models**: Logistic Regression, KNN (k=5), Gaussian Naive Bayes
7. **Feature engineering**: added `DTI_Ratio²` and `Credit_Score²`, then re-evaluated

Precision was treated as the key metric, because a high-risk applicant should not be approved.

## Results (test set, 200 rows)

| Model | Stage | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | Baseline | 0.865 | 0.783 | 0.770 | 0.777 |
| KNN | Baseline | 0.760 | 0.627 | 0.525 | 0.571 |
| Naive Bayes | Baseline | 0.865 | 0.804 | 0.738 | 0.769 |
| Logistic Regression | + Feature engineering | 0.880 | 0.785 | 0.836 | 0.810 |
| KNN | + Feature engineering | 0.785 | 0.673 | 0.574 | 0.619 |
| Naive Bayes | + Feature engineering | 0.860 | 0.811 | 0.705 | 0.754 |

Feature engineering improved Logistic Regression and KNN. Logistic Regression is the best overall after it.

## Tech Stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook

## How to Run
```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
# add loan_approval_data.csv to this folder
jupyter notebook Credit_wise.ipynb
```

## Files
- `Credit_wise.ipynb`: full workflow from cleaning to model evaluation

## Author
Arun
