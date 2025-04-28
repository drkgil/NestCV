## 🧪 Study Purpose 

#### Machine Learning Models Predict Treatment Outcome from Serum Proteins in Patients with Myasthenia Gravis that received Thymectomy

Myasthenia Gravis (MG) is a rare autoimmune disease that causes muscle weakness. Approximately 80–85% of patients have antibodies against acetylcholine receptors (AChRs).
This study aimed to identify serum proteins in MG patients that can predict improvement following thymectomy, based on reductions in the Quantitative MG (QMG) score.

To achieve this, we applied **nested cross-validation (nCV)** to a proteomic dataset of 86 MG patients. nCV is particularly well-suited for smaller datasets, as it helps prevent overfitting by ensuring each sample is independently tested from the data used to train the model.

## 💻 Machine Learning Models Used 

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

# 🚀 Quickstart
Once you've downloaded or cloned the repository:

**1. Open the NestCV R Project**

Open the .Rproj file in RStudio. This automatically sets the working directory for you.

**2. Install Required Packages**

Run the following R code to install the necessary packages:
```r 
# Install and load all required packages
load_packages()
```

**3. Prepare Your Data**

This line of code creates an object named "patientFeaturesData" that will be used by the package. 
 
Copy and use as is, or your substitute your own data for the arguments: 
 
 - data = Data frame or matrix containing features of interest
 - annotation = A binary/categorical vector
 - gene_list = an optional character vector of features to use (in this study, gene names are used), otherwise all data in 'data' will be used.

```r
# these objects are defined in the package and can be used as an example!

prepare_patient_data(
  data = ettx_Binary, 
  annotation = ettx_annotation, 
  gene_list = ETTX_genes
)

```
Running `prepare_patient_data()` will create global variables (data_X, data_Y, patientFeaturesData) that are _**required**_ for the other functions to work.


**4. Final Setup** 

Copy and paste the following code into your R console to run the initial setup:

```r
# Generate all objects necessary for saving results from the nCV 
run_setup()
```

**5. Run nCV!**

Simply copy and paste the following code into your R console. 

```r
# Takes ~ 1 hour to complete using the provided example code.
run_nested_cv()
```

*** 

### Code Usage

This pipeline is designed to predict **binary classification outcomes** from proteomic data.

>⚠ Models are currently configured for binary classification only. While extension to multi-class classification may be possible, it has not been tested and could require minor code modifications.


*** 

### 📥 How to Download or Clone

You can download the project files in two ways:

**Option 1: Download as a ZIP**
- Click the green **Code** button at the top of this page.
- Choose **Download ZIP**.
- Extract the files to a folder on your computer.
- Open `NestCV.Rproj` in RStudio.



**Option 2: Clone using Git**
- Open your terminal or Git Bash and run:
```bash
git clone https://github.com/drkgil/NestCV
```
