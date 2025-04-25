# 🧪 Study Purpose 

#### Machine Learning Models Predict Treatment Outcome from Serum Proteins in Patients with Myasthenia Gravis that received Thymectomy

Myasthenia Gravis (MG) is a rare autoimmune disease that causes muscle weakness. Approximately 80–85% of patients have antibodies against acetylcholine receptors (AChRs).
This study aimed to identify serum proteins in MG patients that can predict improvement following thymectomy, based on reductions in the Quantitative MG (QMG) score.

To achieve this, we applied **nested cross-validation (nCV)** to a proteomic dataset of 86 MG patients. nCV is particularly well-suited for smaller datasets, as it helps prevent overfitting by ensuring each sample is independently tested from the data used to train the model.

# 💻 Machine Learning Models Used 

This code uses the following machine learning models in a **5-fold nested cross-validation** framework:

- Naïve Bayes (NB)
- Logistic Regression (Log)
- Least Absolute Shrinkage and Selection Operator (LASSO)
- Elastic Net (EN)
- Random Forest (RF)
- Classification Tree (CT or classTree)
- Adaptive Boosting (adaBoost)
- Extreme Gradient Boosting (XGBoost)

The inner loop performs hyperparameter tuning via **grid search**, using the `caret` package with its default 10-fold CV. The best parameters for each outer fold are recorded and used to evaluate model performance and select the final model.

# Code Usage

This pipeline is designed to predict **binary classification outcomes** from proteomic data.

>⚠ Models are currently configured for binary classification only. While extension to multi-class classification may be possible, it has not been tested and could require minor code modifications.


*insert here the information and code on how to load the package/repo*

📌 **To get started**, copy and paste the following code into your R console to load packages, prepare your data, and run the initial setup:

```r
# This is example R code
load_packages()

prepare_patient_data(ettx_Binary, ettx_annotation, ETTX_genes)

run_setup()
```

