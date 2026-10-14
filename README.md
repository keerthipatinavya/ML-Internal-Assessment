# ML Internal Assessment — Classification, Clustering & Boosting

## Dataset mapping
1. **Mall_Customers.csv** → K-Means Clustering
2. **Social_Network_Ads.csv** → Classification using Logistic Regression
3. **insurance_pre.csv** → Regression using Gradient Boosting Regressor

## Why these mappings?
- Mall Customers has no labelled target and is designed for customer segmentation, so clustering is appropriate.
- Social Network Ads has a binary `Purchased` target (0/1), so classification is appropriate.
- Insurance has continuous `charges`, so a regression problem is appropriate. Gradient Boosting is used for the required boosting component.

## How to run

```bash
python -m pip install -r requirements.txt
python ml_ia_assignment.py
```

The script creates a `results/` directory containing:
- clustering elbow and silhouette plots
- customer cluster visualization
- cluster profile
- classification metrics and confusion matrix
- boosting metrics and actual-vs-predicted plot
- overall results table

## Important preprocessing decisions
- Identifier columns (`CustomerID`, `User ID`) are excluded because they do not represent predictive/segmentation information.
- Categorical variables are one-hot encoded.
- Classification numerical features are standardized.
- No missing-value imputation is required because all three supplied datasets contain zero missing values.

## Algorithms
### Clustering
K-Means with k selected using silhouette analysis.

### Classification
Logistic Regression with stratified 80/20 train-test split.

### Boosting
Gradient Boosting Regressor with 200 estimators, learning rate 0.05 and maximum tree depth 3.

## GitHub commands

```bash
git init
git add .
git commit -m "Add ML IA datasets and implementations"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```
