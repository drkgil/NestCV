#' @title Initialize objects for downstream analysis
#' @description Prepares necessary objects assuming data_X and data_Y already exist.
#' @export
run_setup <- function() {

  # Initialize training settings for the 'caret' package
  train_control <- trainControl(
    method = "cv",
    returnData = TRUE,
    classProbs = TRUE,
    savePredictions = "final"
  )
  assign("train_control", train_control, envir = .GlobalEnv)

  # Outer loop: create 5 folds for the 80:20 outer split
  #set.seed(123)
  #outer_folds <- createFolds(data_Y[, 1], k = 5)
  #assign("outer_folds", outer_folds, envir = .GlobalEnv)

  # Outer loop: create 5 folds for the 80:20 outer split
  set.seed(123)

  # Function to check if each fold has at least one of each class
  #is_valid_fold <- function(fold_indices, labels) {
   # test_labels <- labels[fold_indices]
   # length(unique(test_labels)) > 1
  #}
  # Corrected function: ensure each outer_test set has both classes
  is_valid_fold <- function(test_indices, labels) {
    test_labels <- labels[test_indices]
    return(length(unique(test_labels)) == 2)
  }

  # Create folds and check if each fold contains both classes
 # repeat {
  #  outer_folds <- createFolds(data_Y[, 1], k = 5)
   # if (all(sapply(outer_folds, is_valid_fold, labels = data_Y[, 1]))) break
  #}

  repeat {
    outer_folds <- createFolds(data_Y[, 1], k = 5)

    # Check if each fold's test set contains both classes
    valid_folds <- all(sapply(outer_folds, is_valid_fold, labels = data_Y[, 1]))

    if (valid_folds) break
  }

  assign("outer_folds", outer_folds, envir = .GlobalEnv)


  # Initialize results storage objects
  outer_results <- vector("list", length(outer_folds))
  outer_inner_results <- vector("list", length(outer_folds))
  the_best_params <- vector("list", length(outer_folds))
  #inner_accuracies <- vector("list", length(inner_folds))
  rf_confusion_array <- vector("list", length(outer_folds))
  #inner_average_accuracy <- vector("list", length(inner_folds))
  nb_roc_perf <- vector("list", length(outer_folds))
  nb_auc_value <- vector("list", length(outer_folds))
  nb_aucpr_value <- vector("list", length(outer_folds))
  nb_pr_perf <-  vector("list", length(outer_folds))
  log_roc_perf <- vector("list", length(outer_folds))
  log_auc_value <- vector("list", length(outer_folds))
  log_aucpr_value <- vector("list", length(outer_folds))
  log_pr_perf <-  vector("list", length(outer_folds))
  lasso_roc_perf <- vector("list", length(outer_folds))
  lasso_auc_value <- vector("list", length(outer_folds))
  lasso_aucpr_value <- vector("list", length(outer_folds))
  lasso_pr_perf <-  vector("list", length(outer_folds))
  en_roc_perf <- vector("list", length(outer_folds))
  en_auc_value <- vector("list", length(outer_folds))
  en_aucpr_value <- vector("list", length(outer_folds))
  en_pr_perf <-  vector("list", length(outer_folds))
  rf_roc_perf <- vector("list", length(outer_folds))
  rf_auc_value <- vector("list", length(outer_folds))
  rf_aucpr_value <- vector("list", length(outer_folds))
  rf_pr_perf <-  vector("list", length(outer_folds))
  ct_roc_perf <- vector("list", length(outer_folds))
  ct_auc_value <- vector("list", length(outer_folds))
  ct_aucpr_value <- vector("list", length(outer_folds))
  ct_pr_perf <-  vector("list", length(outer_folds))
  ada_roc_perf <- vector("list", length(outer_folds))
  ada_auc_value <- vector("list", length(outer_folds))
  ada_aucpr_value <- vector("list", length(outer_folds))
  ada_pr_perf <-  vector("list", length(outer_folds))
  xgb_roc_perf <- vector("list", length(outer_folds))
  xgb_auc_value <- vector("list", length(outer_folds))
  xgb_aucpr_value <- vector("list", length(outer_folds))
  xgb_pr_perf <-  vector("list", length(outer_folds))
  auc_p_values_melted <- vector("list", length(outer_folds))
  nb_imp_df_reordered <- vector("list", length(outer_folds))
  log_coef_df <- vector("list", length(outer_folds))
  lasso_non_zero_coefs <- vector("list", length(outer_folds))
  en_non_zero_coefs_all <- vector("list", length(outer_folds))
  rf_nonzero_importance <- vector("list", length(outer_folds))
  ct_imp_df <- vector("list", length(outer_folds))
  ada_feature_importance_sorted <- vector("list", length(outer_folds))
  xgb_imp <- vector("list", length(outer_folds))
  combined_features_all <- vector("list", length(outer_folds))
  imp_feature_count_df <- vector("list", length(outer_folds))
  inner_variable_importance <- vector("list", length(outer_folds))
  #varImp_models <- vector("list", length(inner_folds))

  nb_roc <- list(
    predictions = vector("list", length(outer_folds)),
    labels = vector("list", length(outer_folds))
  )

  log_roc <- list(
    predictions = vector("list", length(outer_folds)),
    labels = vector("list", length(outer_folds))
  )

  lasso_roc <- list(
    predictions = vector("list", length(outer_folds)),
    labels = vector("list", length(outer_folds))
  )

  en_roc <- list(
    predictions = vector("list", length(outer_folds)),
    labels = vector("list", length(outer_folds))
  )

  rf_roc <- list(
    predictions = vector("list", length(outer_folds)),
    labels = vector("list", length(outer_folds))
  )

  ct_roc <- list(
    predictions = vector("list", length(outer_folds)),
    labels = vector("list", length(outer_folds))
  )

  ada_roc <- list(
    predictions = vector("list", length(outer_folds)),
    labels = vector("list", length(outer_folds))
  )

  xgb_roc <- list(
    predictions = vector("list", length(outer_folds)),
    labels = vector("list", length(outer_folds))
  )

  # for final performance table
  all_metrics <- vector("list", length(outer_folds))

  # Initialize a list to store best parameters for each fold
  outer_fold_params <- vector("list", length(outer_folds))
  #inner_fold_params <- vector("list", length(inner_folds))

  # Assign all objects to the global environment
  assign("outer_results", outer_results, envir = .GlobalEnv)
  assign("outer_inner_results", outer_inner_results, envir = .GlobalEnv)
  assign("the_best_params", the_best_params, envir = .GlobalEnv)
  #assign("inner_accuracies", inner_accuracies, envir = .GlobalEnv)
  assign("rf_confusion_array", rf_confusion_array, envir = .GlobalEnv)
  #assign("inner_average_accuracy", inner_average_accuracy, envir = .GlobalEnv)
  assign("nb_roc_perf", nb_roc_perf, envir = .GlobalEnv)
  assign("nb_auc_value", nb_auc_value, envir = .GlobalEnv)
  assign("nb_aucpr_value", nb_aucpr_value, envir = .GlobalEnv)
  assign("nb_pr_perf", nb_pr_perf, envir = .GlobalEnv)
  assign("log_roc_perf", log_roc_perf, envir = .GlobalEnv)
  assign("log_auc_value", log_auc_value, envir = .GlobalEnv)
  assign("log_aucpr_value", log_aucpr_value, envir = .GlobalEnv)
  assign("log_pr_perf", log_pr_perf, envir = .GlobalEnv)
  assign("lasso_roc_perf", lasso_roc_perf, envir = .GlobalEnv)
  assign("lasso_auc_value", lasso_auc_value, envir = .GlobalEnv)
  assign("lasso_aucpr_value", lasso_aucpr_value, envir = .GlobalEnv)
  assign("lasso_pr_perf", lasso_pr_perf, envir = .GlobalEnv)
  assign("en_roc_perf", en_roc_perf, envir = .GlobalEnv)
  assign("en_auc_value", en_auc_value, envir = .GlobalEnv)
  assign("en_aucpr_value", en_aucpr_value, envir = .GlobalEnv)
  assign("en_pr_perf", en_pr_perf, envir = .GlobalEnv)
  assign("rf_roc_perf", rf_roc_perf, envir = .GlobalEnv)
  assign("rf_auc_value", rf_auc_value, envir = .GlobalEnv)
  assign("rf_aucpr_value", rf_aucpr_value, envir = .GlobalEnv)
  assign("rf_pr_perf", rf_pr_perf, envir = .GlobalEnv)
  assign("ct_roc_perf", ct_roc_perf, envir = .GlobalEnv)
  assign("ct_auc_value", ct_auc_value, envir = .GlobalEnv)
  assign("ct_aucpr_value", ct_aucpr_value, envir = .GlobalEnv)
  assign("ct_pr_perf", ct_pr_perf, envir = .GlobalEnv)
  assign("ada_roc_perf", ada_roc_perf, envir = .GlobalEnv)
  assign("ada_auc_value", ada_auc_value, envir = .GlobalEnv)
  assign("ada_aucpr_value", ada_aucpr_value, envir = .GlobalEnv)
  assign("ada_pr_perf", ada_pr_perf, envir = .GlobalEnv)
  assign("xgb_roc_perf", xgb_roc_perf, envir = .GlobalEnv)
  assign("xgb_auc_value", xgb_auc_value, envir = .GlobalEnv)
  assign("xgb_aucpr_value", xgb_aucpr_value, envir = .GlobalEnv)
  assign("xgb_pr_perf", xgb_pr_perf, envir = .GlobalEnv)
  assign("auc_p_values_melted", auc_p_values_melted, envir = .GlobalEnv)
  assign("nb_imp_df_reordered", nb_imp_df_reordered, envir = .GlobalEnv)
  assign("log_coef_df", log_coef_df, envir = .GlobalEnv)
  assign("lasso_non_zero_coefs", lasso_non_zero_coefs, envir = .GlobalEnv)
  assign("en_non_zero_coefs_all", en_non_zero_coefs_all, envir = .GlobalEnv)
  assign("rf_nonzero_importance", rf_nonzero_importance, envir = .GlobalEnv)
  assign("ct_imp_df", ct_imp_df, envir = .GlobalEnv)
  assign("ada_feature_importance_sorted", ada_feature_importance_sorted, envir = .GlobalEnv)
  assign("xgb_imp", xgb_imp, envir = .GlobalEnv)
  assign("combined_features_all", combined_features_all, envir = .GlobalEnv)
  assign("imp_feature_count_df", imp_feature_count_df, envir = .GlobalEnv)
  assign("inner_variable_importance", inner_variable_importance, envir = .GlobalEnv)
  #assign("varImp_models", varImp_models, envir = .GlobalEnv)

  assign("nb_roc", nb_roc, envir = .GlobalEnv)
  assign("log_roc", log_roc, envir = .GlobalEnv)
  assign("lasso_roc", lasso_roc, envir = .GlobalEnv)
  assign("en_roc", en_roc, envir = .GlobalEnv)
  assign("rf_roc", rf_roc, envir = .GlobalEnv)
  assign("ct_roc", ct_roc, envir = .GlobalEnv)
  assign("ada_roc", ada_roc, envir = .GlobalEnv)
  assign("xgb_roc", xgb_roc, envir = .GlobalEnv)

  assign("all_metrics", all_metrics, envir = .GlobalEnv)

  assign("outer_fold_params", outer_fold_params, envir = .GlobalEnv)
  #assign("inner_fold_params", inner_fold_params, envir = .GlobalEnv)

  message("Setup complete: outer folds result storage objects initialized.")
}
