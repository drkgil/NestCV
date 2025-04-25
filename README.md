---
title: "Information about NestCV"
author: "Karli Gilbert"
date: "2025-04-25"
output: 
  html_document: 
    keep_md: true
---

# Study Purpose 

#### Machine Learning Models Predict Treatment Outcome from Serum Proteins in Patients with Myasthenia Gravis that received Thymectomy

Myasthenia Gravis (MG) is a rare autoimmune disease that causes muscle weakness in which 80-85% of patients have antibodies against acetylcholine receptors (AChRs). Our study aimed to determine if we can identify a set of proteins in MG patient serum that could predict improvement based on Quantitative MG (QMG) score reduction. To efficiently accomplish this goal, we decided to use nested cross validation on a proteomic dataset of 86 patients with MG. Nested cross validation (nCV) is a great method to use on smaller datasets because it prevents overfitting of algorithms by ensuring each sample is tested independently from the data that trained the algorithm. 

This code utilizes the following machine learning models in a 5-fold nested cross-validation method:

- Naïve Bayes (NB)
- Logistic Regression (Log)
- Least Absolute Shrinkage and Selection Operator (LASSO)
- Elastic Net (EN)
- Random Forest (RF)
- Classification Tree (CT or classTree)
- Adaptive Boosting (adaBoost)
- Extreme Gradient Boosting (XGBoost)

The inner loop performs hyperparameter tuning for the respective outer fold training data using a grid search. This grid search is currently defined for each model, but can be easily edited to suit future needs. This training is conducted by the 'caret' package using the default 10-fold cross validation method to pick the best parameters, i.e. the parameters that result in the highest accuracy for the model. The parameters used in each outer-fold, termed the best parameters, are recorded for reference in choosing the best final model. 

# Code Usage
NestCV is designed to perform cross validation with ML models to predict binary categorization of an outcome of interest. Please note that all models are currently configured to run classification predictions and not regressions; it should be possible to run this code to predict 3 or more classifications, but this has not been tested and may require slight adjustments to the model(s) code.



