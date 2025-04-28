load_packages <- function() {
  # Initialize summary_table as NULL
  summary_table <- NULL
  # List of CRAN packages
  cran_packages <- c("knitr", "kableExtra", "dplyr", "ggplot2", "tidyr", "janitor",
                     "ggalluvial", "xgboost", "rBayesianOptimization", "Amelia",
                     "patchwork", "SHAPforxgboost", "tidyquant", "tidyverse",
                     "caret", "PRROC", "viridis", "randomForest", "JOUSBoost",
                     "rpart", "naivebayes", "psych", "ggsignif", "pROC", "gridExtra",
                     "glmnet", "pheatmap", "ppcor", "reshape2", "kernelshap",
                     "RColorBrewer")

  # For custom packages not on CRAN (e.g., shapviz)
  github_packages <- c("shapviz")

  # Install CRAN packages
  for (pkg in cran_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
    # Try loading the package
    tryCatch({
      library(pkg, character.only = TRUE)
    }, error = function(e) {
      message(paste("Package", pkg, "could not be loaded:", e$message))
    })
  }

  # Install GitHub packages
  for (pkg in github_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      tryCatch({
        devtools::install_github(pkg)
      }, error = function(e) {
        message(paste("GitHub package", pkg, "could not be installed:", e$message))
      })
    }
    # Try loading the package
    tryCatch({
      library(pkg, character.only = TRUE)
    }, error = function(e) {
      message(paste("Package", pkg, "could not be loaded:", e$message))
    })
  }
}
