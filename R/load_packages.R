#' @title Load required packages
#' @description Loads packages listed in the `cran_packages` object.
#' @export

load_packages <- function() {
  # List of CRAN packages to install and load
  cran_packages <- c("knitr", "kableExtra", "dplyr", "ggplot2", "tidyr", "janitor",
                     "ggalluvial", "xgboost", "rBayesianOptimization", "Amelia",
                     "patchwork", "SHAPforxgboost", "tidyquant", "tidyverse",
                     "caret", "PRROC", "viridis", "randomForest", "JOUSBoost",
                     "rpart", "naivebayes", "psych", "ggsignif", "pROC", "gridExtra",
                     "glmnet", "pheatmap", "ppcor", "reshape2", "shapviz", "kernelshap",
                     "RColorBrewer")

  # Loop over the package list
  for (pkg in cran_packages) {
    # If the package is not installed, install it
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }

    # Try loading the package and catch any errors
    tryCatch({
      library(pkg, character.only = TRUE)
    }, error = function(e) {
      message(paste("Package", pkg, "could not be loaded:", e$message))
    })
  }
}

