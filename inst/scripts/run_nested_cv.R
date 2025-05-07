# outer progress bar start
pbo = txtProgressBar(min = 0, max = length(outer_folds), initial = 0)

for (i in seq_along(outer_folds)) {
  # Outer train/test split
  test_indices <- outer_folds[[i]]
  outer_train <- patientFeaturesData[-test_indices, ]
  outer_test <- patientFeaturesData[test_indices, ]

  # Remove rows with NA values before processing
  outer_train <- na.omit(outer_train)
  outer_test <- na.omit(outer_test)

  # Make them VALID factors
  X_outer_test <- as.matrix(outer_test[, -ncol(outer_test)])
  Y_outer_test  <- factor(outer_test$Survival_death, levels = c(0,1))  # ensure that there are always 2 levels, regardless of how the data splits!
  X_outer_train <- as.matrix(outer_train[, -ncol(outer_train)])
  Y_outer_train <- factor(outer_train$Survival_death)
  Y_outer_train <- as.numeric(Y_outer_train) -1


  # Inner loop (nested 4-fold CV)
  inner_folds <- createFolds(outer_train$Survival_death, k = 4, list = TRUE)

  # Initialize objects
  inner_fold_params <- vector("list", length(inner_folds))
  #assign("inner_fold_params", inner_fold_params, envir = .GlobalEnv)
  inner_accuracies <- vector("list", length(inner_folds))
  #assign("inner_accuracies", inner_accuracies, envir = .GlobalEnv)
  inner_average_accuracy <- vector("list", length(inner_folds))
  #assign("inner_average_accuracy", inner_average_accuracy, envir = .GlobalEnv)
  varImp_models <- vector("list", length(inner_folds))
  #assign("varImp_models", varImp_models, envir = .GlobalEnv)

  # Start of inner loop

  for (j in seq_along(inner_folds)) {
    # outer progress bar start
    pbi = txtProgressBar(min = 0, max = length(inner_folds), initial = 0)

    inner_test_indices <- inner_folds[[j]]
    inner_train <- outer_train[-inner_test_indices, ]
    inner_test <- outer_train[inner_test_indices, ]

    inner_train$Survival_death <- factor(inner_train$Survival_death, levels = c(0, 1), labels = c("Class0", "Class1"))
    inner_test$Survival_death <- factor(inner_test$Survival_death, levels = c(0, 1), labels = c("Class0", "Class1"))

    X_inner_train <- as.matrix(inner_train[, -ncol(inner_train)])
    Y_inner_train <- as.factor(inner_train$Survival_death)

    X_inner_test <- as.matrix(inner_test[, -ncol(inner_test)])
    Y_inner_test <- as.factor(inner_test$Survival_death)


    # Naive Bayes model training and evaluation
    nb_model <- caret::train(
      x = X_inner_train,
      y = Y_inner_train,
      method = "nb",  # 'nb' for Naive Bayes
      trControl = train_control
    )
    nb_inner_predictions <- predict(nb_model, X_inner_test)


    # Logistic Regression model training and evaluation - no tuning parameters
    log_model <- caret::train(
      x = X_inner_train,
      y = Y_inner_train,
      method = "glm", #'glm' for Logsitic Regression
      trControl = train_control
    )
    log_inner_predictions <- predict(log_model, X_inner_test)


    # LASSO model training and evaluation
    lasso_model <- caret::train(
      x = X_inner_train,
      y = Y_inner_train,
      method = "glmnet",  # 'glmnet' for LASSO
      trControl = train_control,
      tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 100)) # alpha = 1 ensures LASSO
    )
    lasso_inner_predictions <- predict(lasso_model, X_inner_test)

    # Elastic Net model training and evaluation
    en_model <- caret::train(
      x = X_inner_train,
      y = Y_inner_train,
      method = "glmnet",  # 'glmnet' for LASSO
      trControl = train_control,
      tuneGrid = expand.grid(
        alpha = c(0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999),
        lambda = c(0.001, 0.01, 0.1, 1, 10)
      ))
    en_inner_predictions <- predict(en_model, X_inner_test)

    # Random Forest model training and evaluation
    rf_model <- caret::train(
      x = X_inner_train,
      y = Y_inner_train,
      method = "rf",  # 'rf' for Random Forest
      trControl = train_control,
      tuneGrid = expand.grid(mtry = c(1, 3, 5, 7, 10)),
    )
    rf_inner_predictions <- predict(rf_model, X_inner_test)


    # Classification Tree model training and evaluation
    classtree_model <- caret::train(
      x = X_inner_train,
      y = Y_inner_train,
      method = "rpart",  # 'rpart' for Classification Tree
      trControl = train_control
    )
    ct_inner_predictions <- predict(classtree_model, X_inner_test)


    # adaBoost model training and evaluation
    adaboost_model <- caret::train(
      x = X_inner_train,
      y = Y_inner_train,
      method = "AdaBoost.M1",  # 'ada' for adaBoost  # 'AdaBoost.M1' for the most similar method to JOUSBoost
      trControl = train_control,
      tuneGrid = expand.grid(
        mfinal = c(10, 50, 100),  # this is the n_rounds
        maxdepth = c(1, 2, 3),
        coeflearn = c('Breiman', 'Freund', 'Zhu'))
    )
    ada_inner_predictions <- predict(adaboost_model, X_inner_test)


    # XGBoost model training and evaluation; you cannot use early_stopping_rounds here so consider xgboost.train instead
    # xgboost_model <- caret::train(
    # x = X_inner_train,
    # y = Y_inner_train,
    # method = "xgbTree", # 'xgbTree' for XGBoost
    # trControl = train_control
    #)

    # Create a parameter grid
    param_grid <- expand.grid(
      nrounds = c(100, 200),  # Number of boosting rounds
      max_depth = c(3, 6, 9),  # Depth of the trees
      eta = c(0.1, 0.3, 0.5),  # Learning rate
      gamma = c(0, 1),  # Minimum loss reduction required to make a further partition
      colsample_bytree = c(0.6, 0.8, 1),  # Subsample ratio of columns for each tree
      min_child_weight = c(1, 5),  # Minimum sum of instance weight (hessian) in a child
      subsample = c(0.6, 0.8, 1)  # Fraction of samples used to train each tree
    )

    # Placeholder for best model and performance tracking
    best_xgb_model <- NULL
    best_xgb_accuracy <- -Inf
    # Prepare Y labels for XGB
    Y_xgb <- as.numeric(factor(Y_inner_train)) - 1

    # Iterate over parameter grid
    for(k in 1:nrow(param_grid)) {
      params <- param_grid[k, ]

      # Set up model parameters
      xgb_params <- list(
        objective = "binary:logistic",
        max_depth = params$max_depth,
        eta = params$eta,
        gamma = params$gamma,
        colsample_bytree = params$colsample_bytree,
        min_child_weight = params$min_child_weight,
        subsample = params$subsample,
        eval_metric = "auc"
      )
      inner_dtrain <- xgb.DMatrix(data = X_inner_train, label = Y_xgb)
      # Train the model
      xgboost_model <- xgb.train(
        data = inner_dtrain,
        params = xgb_params,
        nrounds = params$nrounds,
        early_stopping_rounds = 10,
        watchlist = list("train" = inner_dtrain),
        verbose = 0  # Turn off verbose output for clarity
      )

      # Track best model based on accuracy
      predictions <- predict(xgboost_model, X_inner_train)
      accuracy <- mean(predictions == Y_xgb)  # Calculate accuracy
      if (accuracy > best_xgb_accuracy) {
        best_xgb_accuracy <- accuracy
        best_xgb_model <- xgboost_model
      }
    }

    # You now have the best_xgb_model with the highest accuracy


    xgb_inner_predictions <- predict(best_xgb_model, X_inner_test)


    # Store multiple metrics for each inner fold
    inner_accuracies[[j]] <- list(
      nb = list (accuracy = mean(nb_inner_predictions == Y_inner_test)),
      log = list(accuracy = mean(log_inner_predictions == Y_inner_test)),
      lasso = list(accuracy = mean(lasso_inner_predictions == Y_inner_test)),
      en = list(accuracy = mean(en_inner_predictions == Y_inner_test)),
      rf = list(accuracy = mean(rf_inner_predictions == Y_inner_test)),
      ct = list(accuracy = mean(ct_inner_predictions == Y_inner_test)),
      ada = list(accuracy = mean(ada_inner_predictions == Y_inner_test)),
      xgb = list(accuracy = mean(xgb_inner_predictions == Y_inner_test))
    )


    # Collect best parameters for each model
    inner_fold_params[[j]] <- list(
      nb = list(usekernel = nb_model$bestTune$usekernel,
                laplace = nb_model$bestTune$laplace),
      lasso = list(
        lambda = lasso_model$bestTune$lambda
      ),
      en = list(
        alpha = en_model$bestTune$alpha,
        lambda = en_model$bestTune$lambda
      ),
      rf = list(mtry = rf_model$bestTune$mtry),
      classtree = list(cp = classtree_model$bestTune$cp),
      adaboost = list(
        mfinal = adaboost_model$bestTune$mfinal,
        maxdepth = adaboost_model$bestTune$maxdepth,
        coeflearn = adaboost_model$bestTune$coeflearn # doesn't really get used but saving because it's given
      ),
      xgboost = list(
        # nrounds = xgboost_model$bestTune$nrounds,
        max_depth = best_xgb_model[["params"]][["max_depth"]],
        eta = best_xgb_model[["params"]][["eta"]],
        gamma = best_xgb_model[["params"]][["gamma"]],
        colsample_bytree = best_xgb_model[["params"]][["colsample_bytree"]],
        min_child_weight = best_xgb_model[["params"]][["min_child_weight"]],
        subsample = best_xgb_model[["params"]][["subsample"]]
      )
    )


    # Variable Importance for all inner loop models but ct
    # Currently only set up to do it for each inner loop but not saved outside of the inner (i.e. overwritten each time)
    # Initialize a list to store variable importances
    varImp_models[[j]] <- list(
      nb = varImp(nb_model)[["importance"]],
      log = varImp(log_model)[["importance"]],
      lasso = varImp(lasso_model)[["importance"]],
      en = varImp(en_model)[["importance"]],
      rf = varImp(rf_model)[["importance"]],
      # ct = varImp(classtree_model)[["importance"]],
      ada = varImp(adaboost_model)[["importance"]],
      xgb = xgb.importance(model = best_xgb_model)
    )


    setTxtProgressBar(pbi,j)

  } # End of Inner Loop

  close(pbi)


  # Store all inner fold accuracies for this outer fold
  outer_inner_results[[i]] <- inner_accuracies

  # Store all inner fold variable importance scores for this outer fold
  inner_variable_importance[[i]] <- varImp_models

  # Inner CV accuracy i.e. average accuracy of all algorithms over all 4-inner-folds per outer fold
  inner_average_accuracy[[i]] <- mean(unlist(inner_accuracies))

  # Store all inner fold params for this outer fold
  outer_fold_params[[i]] <- inner_fold_params


  # Initialize an empty list to store data frames for each fold
  inner_results_df <- lapply(seq_along(inner_accuracies), function(j) {
    fold_results <- inner_accuracies[[j]]
    data.frame(
      fold = j,
      nb_accuracy = fold_results$nb$accuracy,
      log_accuracy = fold_results$log$accuracy,
      lasso_accuracy = fold_results$lasso$accuracy,
      en_accuracy = fold_results$en$accuracy,
      rf_accuracy = fold_results$rf$accuracy,
      ct_accuracy = fold_results$ct$accuracy,
      ada_accuracy = fold_results$ada$accuracy,
      xgb_accuracy = fold_results$xgb$accuracy
    )
  })

  # Combine all folds into a single data frame
  inner_results_df <- do.call(rbind, inner_results_df)



  # Find the fold with the highest accuracy for each model
  best_folds <- inner_results_df %>%
    summarise(
      best_nb = fold[which.max(nb_accuracy)],
      best_log = fold[which.max(log_accuracy)], # will be NULL
      best_lasso = fold[which.max(lasso_accuracy)],
      best_en = fold[which.max(en_accuracy)],
      best_rf = fold[which.max(rf_accuracy)],
      best_classtree = fold[which.max(ct_accuracy)],
      best_adaboost = fold[which.max(ada_accuracy)],
      best_xgboost = fold[which.max(xgb_accuracy)]
    )

  # Extract the best parameters based on the best performing folds
  best_params <- lapply(names(best_folds), function(model) {
    fold_index <- best_folds[[model]]
    inner_fold_params[[fold_index]][[gsub("best_", "", model)]]
  })

  names(best_params) <- gsub("best_", "", names(best_folds))

  the_best_params[[i]] <- best_params


  # Final outer loop models

  nb_outer_model <- e1071::naiveBayes(factor(Survival_death) ~ ., data = outer_train, usekernel = best_params$nb$usekernel, laplace = best_params$nb$laplace)
  log_outer_model <- glm(Survival_death ~ ., data = outer_train, family = "binomial")
  lasso_outer_fit <- glmnet(X_outer_train, Y_outer_train, family = "binomial")
  set.seed(123)
  lasso_outer_model <- cv.glmnet(X_outer_train, Y_outer_train)
  set.seed(123)
  en_outer_model <- cv.glmnet(X_outer_train, Y_outer_train, alpha = best_params$en$alpha)
  rf_outer_model <- randomForest::randomForest(factor(Survival_death) ~., data = outer_train, importance = TRUE, mtry = best_params$rf$mtry)
  # ct_outer_model <- rpart(factor(Survival_death) ~ ., data = outer_train, method = "class", control = rpart.control(cp = best_params$classtree$cp))
  ct_outer_model <- rpart(factor(Survival_death) ~ ., data = outer_train, method = "class")
  ada_outer_Y <- ifelse(Y_outer_train == 0, -1, 1)
  ada_outer_model <- adaboost(X_outer_train, ada_outer_Y, verbose = FALSE, control = rpart::rpart.control(maxdepth = best_params$adaboost$maxdepth), n_rounds = best_params$adaboost$mfinal)
  dtrain <- xgb.DMatrix(data = X_outer_train, label = Y_outer_train)
  params <- list("max_depth" = best_params$xgboost$max_depth, "eta" = best_params$xgboost$eta, "gamma" = best_params$xgboost$gamma, "colsample_bytree" = best_params$xgboost$colsample_bytree, "min_child_weight" = best_params$xgboost$min_child_weight, "subsample" = best_params$xgboost$subsample, "objective"="binary:logistic", "eval_metric"= "auc")
  xgb_outer_model <- xgb.train(params = params, data = dtrain, watchlist = list("train" = dtrain), early_stopping_rounds = 10, nrounds = 100, verbose = 0)

  # Outer test set predictions

  # Naive Bayes
  nb_outer_predictions <- predict(nb_outer_model, X_outer_test)

  # Logistic Regression
  log_outer_predictions <- predict(log_outer_model, data.frame(X_outer_test))
  log_outer_predictions = ifelse(log_outer_predictions > 0.5, 1, 0)

  # LASSO
  lasso_outer_predictions <- predict(lasso_outer_model, newx = X_outer_test, s = "lambda.min")
  lasso_outer_predictions <- ifelse(lasso_outer_predictions > 0.5, 1, 0)

  # Elastic Net
  en_outer_predictions <- predict(en_outer_model, newx = X_outer_test, s = "lambda.min")
  en_outer_predictions <- ifelse(en_outer_predictions > 0.5, 1, 0)

  # Random Forest
  rf_outer_predictions <- predict(rf_outer_model, X_outer_test)

  # Classification Tree
  ct_outer_predictions <- predict(ct_outer_model, data.frame(X_outer_test), type = "class")

  # adaBoost
  ada_outer_predictions <- predict(ada_outer_model, X_outer_test)
  ada_outer_predictions <- ifelse(ada_outer_predictions == -1, 0, 1)

  # XGBoost
  dtest <- xgb.DMatrix(data = X_outer_test)
  xgb_outer_predictions <- predict(object = xgb_outer_model, newdata = dtest)
  xgb_outer_predictions <- ifelse(xgb_outer_predictions > 0.5, 1, 0)

  # Confusion Matrix
  con_NB <- confusionMatrix(nb_outer_predictions, Y_outer_test, positive = "1")
  con_Logistic <- confusionMatrix(factor(log_outer_predictions), Y_outer_test, positive = "1")
  con_lasso <- confusionMatrix(as.factor(lasso_outer_predictions), Y_outer_test, positive = "1")
  con_en <- confusionMatrix(as.factor(en_outer_predictions), Y_outer_test, positive = "1")
  con_RF <- confusionMatrix(rf_outer_predictions, Y_outer_test, positive = "1")
  con_classTree <- confusionMatrix(ct_outer_predictions, Y_outer_test, positive = "1")
  con_adaBoost <- confusionMatrix(factor(ada_outer_predictions), Y_outer_test, positive = "1")
  con_XGB <- confusionMatrix(factor(xgb_outer_predictions), Y_outer_test, positive = "1")

  MCC_NB <- list(
    TP <- con_NB$table[[4]],
    FP <- con_NB$table[[2]],
    FN <- con_NB$table[[3]],
    TN <- con_NB$table[[1]]
  ) %>%  # # MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
    pmap_dbl(., ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))

  NB_pred_outcome <- cbind(as.numeric(nb_outer_predictions)-1, as.numeric(Y_outer_test)-1) %>%
    data.frame() %>%
    setNames(c("predictions", "outcome"))

  NB_fg <- NB_pred_outcome %>%
    filter(outcome == 1) %>%
    pull(predictions)

  NB_bg <- NB_pred_outcome %>%
    filter(outcome == 0) %>%
    pull(predictions)


  MCC_Logistic <- list(
    TP <- con_Logistic$table[[4]],
    FP <- con_Logistic$table[[2]],
    FN <- con_Logistic$table[[3]],
    TN <- con_Logistic$table[[1]]
  ) %>%  # # MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
    pmap_dbl(., ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))

  Logistic_pred_outcome <- cbind(as.numeric(log_outer_predictions), as.numeric(Y_outer_test)-1) %>%
    data.frame() %>%
    setNames(c("predictions", "outcome"))

  Logistic_fg <- Logistic_pred_outcome %>%
    filter(outcome == 1) %>%
    drop_na() %>%
    pull(predictions)

  Logistic_bg <- Logistic_pred_outcome %>%
    filter(outcome == 0) %>%
    drop_na() %>%
    pull(predictions)

  MCC_Lasso <- list(
    TP <- con_lasso$table[[4]],
    FP <- con_lasso$table[[2]],
    FN <- con_lasso$table[[3]],
    TN <- con_lasso$table[[1]]
  ) %>%  # # MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
    pmap_dbl(., ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))

  Lasso_pred_outcome <- cbind(as.numeric(lasso_outer_predictions), as.numeric(Y_outer_test)-1) %>%
    data.frame() %>%
    setNames(c("predictions", "outcome"))

  Lasso_fg <- Lasso_pred_outcome %>%
    filter(outcome == 1) %>%
    pull(predictions)

  Lasso_bg <- Lasso_pred_outcome %>%
    filter(outcome == 0) %>%
    pull(predictions)

  MCC_En <- list(
    TP <- con_en$table[[4]],
    FP <- con_en$table[[2]],
    FN <- con_en$table[[3]],
    TN <- con_en$table[[1]]
  ) %>%  # # MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
    pmap_dbl(., ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))

  En_pred_outcome <- cbind(as.numeric(en_outer_predictions), as.numeric(Y_outer_test)-1) %>%
    data.frame() %>%
    setNames(c("predictions", "outcome"))

  En_fg <- En_pred_outcome %>%
    filter(outcome == 1) %>%
    pull(predictions)

  En_bg <- En_pred_outcome %>%
    filter(outcome == 0) %>%
    pull(predictions)


  MCC_RF <- list(
    TP <- con_RF$table[[4]],
    FP <- con_RF$table[[2]],
    FN <- con_RF$table[[3]],
    TN <- con_RF$table[[1]]
  ) %>%  # # MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
    pmap_dbl(., ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))

  RF_pred_outcome <- cbind(as.numeric(rf_outer_predictions)-1, as.numeric(Y_outer_test)-1) %>%
    data.frame() %>%
    setNames(c("predictions", "outcome"))

  RF_fg <- RF_pred_outcome %>%
    filter(outcome == 1) %>%
    drop_na() %>%
    pull(predictions)

  RF_bg <- RF_pred_outcome %>%
    filter(outcome == 0) %>%
    drop_na() %>%
    pull(predictions)


  MCC_classTree <- list(
    TP <- con_classTree$table[[4]],
    FP <- con_classTree$table[[2]],
    FN <- con_classTree$table[[3]],
    TN <- con_classTree$table[[1]]
  ) %>%  # # MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
    pmap_dbl(., ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))

  classTree_pred_outcome <- cbind(as.numeric(ct_outer_predictions), as.numeric(Y_outer_test)-1) %>%
    data.frame() %>%
    setNames(c("predictions", "outcome"))

  classTree_fg <- classTree_pred_outcome %>%
    filter(outcome == 1) %>%
    pull(predictions)

  classTree_bg <- classTree_pred_outcome %>%
    filter(outcome == 0) %>%
    pull(predictions)

  MCC_adaBoost <- list(
    TP <- con_adaBoost$table[[4]],
    FP <- con_adaBoost$table[[2]],
    FN <- con_adaBoost$table[[3]],
    TN <- con_adaBoost$table[[1]]
  ) %>%  # # MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
    pmap_dbl(., ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))

  adaBoost_pred_outcome <- cbind(as.numeric(ada_outer_predictions), as.numeric(Y_outer_test)-1) %>%
    data.frame() %>%
    setNames(c("predictions", "outcome"))

  adaBoost_fg <- adaBoost_pred_outcome %>%
    filter(outcome == 1) %>%
    pull(predictions)

  adaBoost_bg <- adaBoost_pred_outcome %>%
    filter(outcome == 0) %>%
    pull(predictions)


  MCC_XGB <- list(
    TP <- con_XGB$table[[4]],
    FP <- con_XGB$table[[2]],
    FN <- con_XGB$table[[3]],
    TN <- con_XGB$table[[1]]
  ) %>%  # # MCC <- ((TP * TN) - (FP * FN)) / sqrt((TP + FP) * (TP+FN) * (TN+FP) * (TN+FN))
    pmap_dbl(., ~ ((..1 * ..4) - (..2 * ..3))/sqrt((..1 + ..2) * (..1 + ..3) * (..4 + ..2) * (..4 + ..3)))

  XGB_pred_outcome <- cbind(as.numeric(xgb_outer_predictions), as.numeric(Y_outer_test)-1) %>%
    data.frame() %>%
    setNames(c("predictions", "outcome"))

  XGB_fg <- XGB_pred_outcome %>%
    filter(outcome == 1) %>%
    pull(predictions)

  XGB_bg <- XGB_pred_outcome %>%
    filter(outcome == 0) %>%
    pull(predictions)


  # ROC and PR results
  # Naive Bayes
  nb_roc_pred_ind <- predict(nb_outer_model, X_outer_test, type = "raw")
  nb_roc$predictions[[i]] <- nb_roc_pred_ind[,2]
  nb_roc$labels[[i]] <- Y_outer_test
  nb_roc_predictions <- predict(nb_outer_model, X_outer_test, type = "raw")
  nb_roc_pred <- prediction(nb_roc_predictions[,2], Y_outer_test)
  nb_roc_perf[[i]] <- performance(nb_roc_pred, "tpr", "fpr")
  nb_auc <- performance(nb_roc_pred, measure = "auc")
  nb_auc_value[[i]] <- nb_auc@y.values[[1]]
  nb_aucpr <- performance(nb_roc_pred, measure = "aucpr")
  nb_aucpr_value[[i]] <- nb_aucpr@y.values[[1]]
  nb_pr_perf[[i]] <- performance(nb_roc_pred, "prec", "rec")

  # Logistic Regression
  log_roc_predictions <- unname(unclass(predict(log_outer_model, data.frame(X_outer_test), type = "response")))
  # Ensure predictions are numeric
  log_roc_prob_clean <- as.numeric(log_roc_predictions)
  # Ensure Y_outer_test is numeric (if it's a factor)
  Y_outer_test_num <- as.numeric(Y_outer_test)
  # Store predictions and labels
  log_roc$predictions[[i]] <- log_roc_prob_clean
  log_roc$labels[[i]] <- Y_outer_test_num
  # Generate the prediction object
  log_roc_pred <- prediction(log_roc_prob_clean, Y_outer_test_num)
  # Calculate performance metrics
  log_roc_perf[[i]] <- performance(log_roc_pred, "tpr", "fpr")
  log_auc <- performance(log_roc_pred, measure = "auc")
  log_auc_value[[i]] <- log_auc@y.values[[1]]
  log_aucpr <- performance(log_roc_pred, measure = "aucpr")
  log_aucpr_value[[i]] <- log_aucpr@y.values[[1]]
  log_pr_perf[[i]] <- performance(log_roc_pred, "prec", "rec")


  # LASSO
  lasso_roc$predictions[[i]] <- predict(lasso_outer_model, newx = X_outer_test, s = "lambda.min")
  lasso_roc$labels[[i]] <- Y_outer_test
  lasso_roc_predictions <- predict(lasso_outer_model, newx = X_outer_test, s = "lambda.min")
  lasso_roc_pred <- prediction(lasso_roc_predictions, Y_outer_test)
  lasso_roc_perf[[i]] <- performance(lasso_roc_pred, "tpr", "fpr")
  lasso_auc <- performance(lasso_roc_pred, measure = "auc")
  lasso_auc_value[[i]] <- lasso_auc@y.values[[1]]
  lasso_aucpr <- performance(lasso_roc_pred, measure = "aucpr")
  lasso_aucpr_value[[i]] <- lasso_aucpr@y.values[[1]]
  lasso_pr_perf[[i]] <- performance(lasso_roc_pred, "prec", "rec")

  # Elastic Net
  en_roc$predictions[[i]] <- predict(en_outer_model, newx = X_outer_test, s = "lambda.min")
  en_roc$labels[[i]] <- Y_outer_test
  en_roc_predictions <- predict(en_outer_model, newx = X_outer_test, s = "lambda.min")
  en_roc_pred <- prediction(en_roc_predictions, Y_outer_test)
  en_roc_perf[[i]] <- performance(en_roc_pred, "tpr", "fpr")
  en_auc <- performance(en_roc_pred, measure = "auc")
  en_auc_value[[i]] <- en_auc@y.values[[1]]
  en_aucpr <- performance(en_roc_pred, measure = "aucpr")
  en_aucpr_value[[i]] <- en_aucpr@y.values[[1]]
  en_pr_perf[[i]] <- performance(en_roc_pred, "prec", "rec")

  # Random Forest
  rf_roc_pred_ind <- predict(rf_outer_model, X_outer_test, type = "prob")
  rf_roc$predictions[[i]] <- rf_roc_pred_ind[,2]
  rf_roc$labels[[i]] <- Y_outer_test
  rf_roc_predictions <- predict(rf_outer_model, X_outer_test, type = "prob")
  rf_roc_pred <- prediction(rf_roc_predictions[,2], Y_outer_test)
  rf_roc_perf[[i]] <- performance(rf_roc_pred, "tpr", "fpr")
  rf_auc <- performance(rf_roc_pred, measure = "auc")
  rf_auc_value[[i]] <- rf_auc@y.values[[1]]
  rf_aucpr <- performance(rf_roc_pred, measure = "aucpr")
  rf_aucpr_value[[i]] <- rf_aucpr@y.values[[1]]
  rf_pr_perf[[i]] <- performance(rf_roc_pred, "prec", "rec")

  # Classification Tree
  ct_roc_pred_ind <- predict(ct_outer_model, data.frame(X_outer_test), type = "prob")
  ct_roc$predictions[[i]] <- ct_roc_pred_ind[,2]
  ct_roc$labels[[i]] <- Y_outer_test
  ct_roc_predictions <- predict(ct_outer_model, data.frame(X_outer_test), type = "prob")
  ct_roc_pred <- prediction(ct_roc_predictions[,2], Y_outer_test)
  ct_roc_perf[[i]] <- performance(ct_roc_pred, "tpr", "fpr")
  ct_auc <- performance(ct_roc_pred, measure = "auc")
  ct_auc_value[[i]] <- ct_auc@y.values[[1]]
  ct_aucpr <- performance(ct_roc_pred, measure = "aucpr")
  ct_aucpr_value[[i]] <- ct_aucpr@y.values[[1]]
  ct_pr_perf[[i]] <- performance(ct_roc_pred, "prec", "rec")

  # adaBoost
  ada_roc$predictions[[i]] <- predict(ada_outer_model, X_outer_test, type = "prob")
  ada_roc$labels[[i]] <- Y_outer_test
  ada_roc_predictions <- predict(ada_outer_model, X_outer_test, type = "prob")
  ada_roc_pred <- prediction(ada_roc_predictions, Y_outer_test)
  ada_roc_perf[[i]] <- performance(ada_roc_pred, "tpr", "fpr")
  ada_auc <- performance(ada_roc_pred, measure = "auc")
  ada_auc_value[[i]] <- ada_auc@y.values[[1]]
  ada_aucpr <- performance(ada_roc_pred, measure = "aucpr")
  ada_aucpr_value[[i]] <- ada_aucpr@y.values[[1]]
  ada_pr_perf[[i]] <- performance(ada_roc_pred, "prec", "rec")

  # XGBoost
  xgb_roc$predictions[[i]] <- predict(xgb_outer_model, dtest, type = "prob")
  xgb_roc$labels[[i]] <- Y_outer_test
  xgb_roc_predictions <- predict(xgb_outer_model, dtest, type = "prob")
  xgb_roc_pred <- prediction(xgb_roc_predictions, Y_outer_test)
  xgb_roc_perf[[i]] <- performance(xgb_roc_pred, "tpr", "fpr")
  xgb_auc <- performance(xgb_roc_pred, measure = "auc")
  xgb_auc_value[[i]] <- xgb_auc@y.values[[1]]
  xgb_aucpr <- performance(xgb_roc_pred, measure = "aucpr")
  xgb_aucpr_value[[i]] <- xgb_aucpr@y.values[[1]]
  xgb_pr_perf[[i]] <- performance(xgb_roc_pred, "prec", "rec")

  # pROC can do statistics comparing 2 auc!!

  test_nb <- pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(nb_roc_pred@predictions)))
  test_log <- pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(log_roc_pred@predictions)))
  test_lasso <- pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(lasso_roc_pred@predictions)))
  test_en <- pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(en_roc_pred@predictions)))
  test_rf <- pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(rf_roc_pred@predictions)))
  test_ct <- pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(ct_roc_pred@predictions)))
  test_ada <- pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(ada_roc_pred@predictions)))
  test_xgb <- pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(xgb_roc_pred@predictions)))

  # Your 8 AUC objects
  auc_models <- list(
    nb = test_nb,
    log = test_log,
    lasso = test_lasso,
    en = test_en,
    rf = test_rf,
    ct = test_ct,
    ada = test_ada,
    xgb = test_xgb
  )

  # Create a matrix to store p-values
  auc_model_names <- names(auc_models)
  p_values <- matrix(NA, nrow = length(auc_models), ncol = length(auc_models), dimnames = list(auc_model_names, auc_model_names))

  # Pairwise ROC tests with additional checks
  for (m in seq_along(auc_models)) {
    for (n in seq_along(auc_models)) {
      if (m < n) {  # Only test pairs once

        # Check for NaN or Inf in AUCs
        if (is.nan(auc_models[[m]]) || is.nan(auc_models[[n]]) ||
            is.infinite(auc_models[[m]]) || is.infinite(auc_models[[n]])) {
          message("Skipping due to NaN/Inf for models: ", auc_model_names[m], " vs ", auc_model_names[n])
          next
        }

        # Skip if both AUCs are 1
        if (auc_models[[m]] == 1 && auc_models[[n]] == 1) {
          message("Skipping due to both AUCs = 1 for models: ", auc_model_names[m], " vs ", auc_model_names[n])
          next
        }

        # Run roc.test with error handling
        test_result <- tryCatch({
          pROC::roc.test(
            as.numeric(Y_outer_test),
            as.numeric(unlist(get(paste0(auc_model_names[m], "_roc_pred"))@predictions)),
            as.numeric(unlist(get(paste0(auc_model_names[n], "_roc_pred"))@predictions))
          )
        }, error = function(e) {
          message("Comparison failed for models: ", auc_model_names[m], " vs ", auc_model_names[n])
          message("Error: ", e$message)
          return(NULL)
        })

        # Save p-value
        if (!is.null(test_result)) {
          p_values[m, n] <- test_result$p.value
          print(paste("Storing p-value for models:", auc_model_names[m], "vs", auc_model_names[n], "p-value:", test_result$p.value))
        } else {
          p_values[m, n] <- NA
        }
      }
    }
  }

  # Convert to a tidy data frame
  p_values_df <- as.data.frame(as.table(p_values))
  p_values_df <- p_values_df[!is.na(p_values_df$Freq), ]
  colnames(p_values_df) <- c("Model1", "Model2", "p.value")

  # Convert p-value matrix to a data frame for plotting
  p_values[lower.tri(p_values, diag = TRUE)] <- NA # Keep only upper triangle
  p_values_melted <- melt(p_values, na.rm = TRUE)
  # Add significance stars based on p-value
  p_values_melted$significance <- with(p_values_melted, ifelse(
    value < 0.001, "***",
    ifelse(value < 0.01, "**",
           ifelse(value < 0.05, "*", ""))
  ))

  # Display p-values or asterisks
  p_values_melted$label <- ifelse(
    p_values_melted$significance != "",
    p_values_melted$significance,  # show stars if significant
    sprintf("%.3f", p_values_melted$value) # show p-value otherwise
  )

  # Save each outer fold iteration of p_values_melted:
  auc_p_values_melted[[i]] <- p_values_melted



  performance_table <- data.frame(
    Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "MCC", "AUC", "AUCPR", "RMSE", "TP", "FP", "FN", "TN"),
    `Naive Bayes` = c(
      con_NB$overall["Accuracy"],
      con_NB$byClass["Sensitivity"],
      con_NB$byClass["Specificity"],
      con_NB$byClass["Precision"],
      con_NB$byClass["F1"],
      MCC_NB,
      pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(nb_roc_pred@predictions))),
      unlist((performance(nb_roc_pred, measure = "aucpr")@y.values)),
      unlist((performance(nb_roc_pred, measure = "rmse")@y.values)),
      con_NB$table[[4]],
      con_NB$table[[2]],
      con_NB$table[[3]],
      con_NB$table[[1]]
    ),
    `Logistic` = c(
      con_Logistic$overall["Accuracy"],
      con_Logistic$byClass["Sensitivity"],
      con_Logistic$byClass["Specificity"],
      con_Logistic$byClass["Precision"],
      con_Logistic$byClass["F1"],
      MCC_Logistic,
      pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(log_roc_pred@predictions))),
      unlist((performance(log_roc_pred, measure = "aucpr")@y.values)),
      unlist((performance(log_roc_pred, measure = "rmse")@y.values)),
      con_Logistic$table[[4]],
      con_Logistic$table[[2]],
      con_Logistic$table[[3]],
      con_Logistic$table[[1]]
    ),
    `LASSO` = c(
      con_lasso$overall["Accuracy"],
      con_lasso$byClass["Sensitivity"],
      con_lasso$byClass["Specificity"],
      con_lasso$byClass["Precision"],
      con_lasso$byClass["F1"],
      MCC_Lasso,
      pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(lasso_roc_pred@predictions))),
      unlist((performance(lasso_roc_pred, measure = "aucpr")@y.values)),
      unlist((performance(lasso_roc_pred, measure = "rmse")@y.values)),
      con_lasso$table[[4]],
      con_lasso$table[[2]],
      con_lasso$table[[3]],
      con_lasso$table[[1]]
    ),
    `EN` = c(
      con_en$overall["Accuracy"],
      con_en$byClass["Sensitivity"],
      con_en$byClass["Specificity"],
      con_en$byClass["Precision"],
      con_en$byClass["F1"],
      MCC_En,
      pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(en_roc_pred@predictions))),
      unlist((performance(en_roc_pred, measure = "aucpr")@y.values)),
      unlist((performance(en_roc_pred, measure = "rmse")@y.values)),
      con_en$table[[4]],
      con_en$table[[2]],
      con_en$table[[3]],
      con_en$table[[1]]
    ),
    `RF` = c(
      con_RF$overall["Accuracy"],
      con_RF$byClass["Sensitivity"],
      con_RF$byClass["Specificity"],
      con_RF$byClass["Precision"],
      con_RF$byClass["F1"],
      MCC_RF,
      pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(rf_roc_pred@predictions))),
      unlist((performance(rf_roc_pred, measure = "aucpr")@y.values)),
      unlist((performance(rf_roc_pred, measure = "rmse")@y.values)),
      con_RF$table[[4]],
      con_RF$table[[2]],
      con_RF$table[[3]],
      con_RF$table[[1]]
    ),
    `Class Tree` = c(
      con_classTree$overall["Accuracy"],
      con_classTree$byClass["Sensitivity"],
      con_classTree$byClass["Specificity"],
      con_classTree$byClass["Precision"],
      con_classTree$byClass["F1"],
      MCC_classTree,
      pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(ct_roc_pred@predictions))),
      unlist((performance(ct_roc_pred, measure = "aucpr")@y.values)),
      unlist((performance(ct_roc_pred, measure = "rmse")@y.values)),
      con_classTree$table[[4]],
      con_classTree$table[[2]],
      con_classTree$table[[3]],
      con_classTree$table[[1]]
    ),
    `adaBoost` = c(
      con_adaBoost$overall["Accuracy"],
      con_adaBoost$byClass["Sensitivity"],
      con_adaBoost$byClass["Specificity"],
      con_adaBoost$byClass["Precision"],
      con_adaBoost$byClass["F1"],
      MCC_adaBoost,
      pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(ada_roc_pred@predictions))),
      unlist((performance(ada_roc_pred, measure = "aucpr")@y.values)),
      unlist((performance(ada_roc_pred, measure = "rmse")@y.values)),
      con_adaBoost$table[[4]],
      con_adaBoost$table[[2]],
      con_adaBoost$table[[3]],
      con_adaBoost$table[[1]]
    ),
    `XGBoost` = c(
      con_XGB$overall["Accuracy"],
      con_XGB$byClass["Sensitivity"],
      con_XGB$byClass["Specificity"],
      con_XGB$byClass["Precision"],
      con_XGB$byClass["F1"],
      MCC_XGB,
      pROC::auc(as.numeric(Y_outer_test), as.numeric(unlist(xgb_roc_pred@predictions))),
      unlist((performance(xgb_roc_pred, measure = "aucpr")@y.values)),
      unlist((performance(xgb_roc_pred, measure = "rmse")@y.values)),
      con_XGB$table[[4]],
      con_XGB$table[[2]],
      con_XGB$table[[3]],
      con_XGB$table[[1]]
    )
  )

  # Store the result for this iteration
  all_metrics[[i]] <- performance_table


  # Random Forest confusion arrays
  rf_confusion_array[[i]] <- rf_outer_model$confusion

  # Store outer results
  outer_results[[i]] <- list(
    actual = Y_outer_test,
    nb_predictions = nb_outer_predictions,
    log_predictions = log_outer_predictions,
    lasso_predictions = lasso_outer_predictions,
    en_predictiosn = en_outer_predictions,
    rf_predictions = rf_outer_predictions,
    ct_predictions = ct_outer_predictions,
    ada_predictions = ada_outer_predictions,
    xgb_predictions = xgb_outer_predictions
  )



  # Feature Importance
  # Naive Bayes

  importance_scores <- sapply(nb_outer_model$tables, function(tbl) {
    if (is.matrix(tbl)) {  # Only process numeric features
      means <- tbl[1, ]  # Mean for each class
      sds <- tbl[2, ]    # SD for each class

      # Avoid division by zero
      sds[sds == 0] <- 1e-6

      # Compute log-likelihood components (ignoring actual x values)
      log_likelihoods <- - (means^2) / (2 * sds^2) - log(sds)

      return(sd(log_likelihoods, na.rm = TRUE))  # SD across classes
    } else {
      return(NA)  # Skip categorical variables
    }
  })

  nb_imp_df <- as.data.frame(importance_scores)

  nb_imp_df$Feature <- rownames(nb_imp_df)

  # Reorder the dataframe by importance_scores
  nb_imp_df <- nb_imp_df[order(nb_imp_df$importance_scores, decreasing = TRUE), ]
  nb_imp_df_reordered[[i]] <- nb_imp_df

  # Extract the reordered feature names from the row names
  nb_imp_features <- nb_imp_df$Feature


  # Logistic Regression
  # Coefficient Plot: Effect of Predictors

  # Coefficients for the logistic regression model
  coef_df <- data.frame(
    Variable = names(coef(log_outer_model)),
    Coefficient = coef(log_outer_model)
  )

  #Removing Intercept as it overshadows the entire plot otherwise
  coef_df <- data.frame(coef_df[-1,])

  # Sorting the coefficients by their absolute value in descending order
  coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), ]
  log_coef_df[[i]] <- coef_df

  # Extract the reordered variable names
  log_imp_features <- coef_df$Variable


  # LASSO
  # Feature Importance by the lambda min coefficient
  # note: lasso_outer_model is the same as cvfit in the rmarkdown file

  # get the value of lambda.min and the model coefficients at that value of λ
  coef_min <- as.matrix(coef(lasso_outer_model, s = "lambda.min"))

  # Convert the coefficients into a data frame
  coef_min_df <- as.data.frame(as.matrix(coef_min))
  selected_coefs <- as.matrix(coef_min[2:nrow(coef_min),]) # get everything but the intercept
  sorted_coefs <- selected_coefs[order(abs(selected_coefs), decreasing = TRUE),]			# alternatively use this for saving & graphing
  sorted_coefs_names <- data.frame(Feature = names(sorted_coefs), Importance = sorted_coefs)

  # Filter rows where Importance is non-zero
  non_zero_coefs <- sorted_coefs_names[sorted_coefs_names$Importance != 0, ]
  lasso_non_zero_coefs[[i]] <- non_zero_coefs

  # Extract Feature names
  lasso_imp_features <- non_zero_coefs$Feature


  # Elastic Net
  # Feature Importance by the lambda min coefficient
  # get the value of lambda.min and the model coefficients at that value of λ
  en_coef_min <- as.matrix(coef(en_outer_model, s = "lambda.min"))

  # Convert the coefficients into a data frame
  en_coef_min_df <- as.data.frame(as.matrix(en_coef_min))
  en_selected_coefs <- as.matrix(en_coef_min[2:nrow(en_coef_min),]) # get everything but the intercept
  en_sorted_coefs <- en_selected_coefs[order(abs(en_selected_coefs), decreasing = TRUE),]
  en_sorted_coefs_names <- data.frame(Feature = names(en_sorted_coefs), Importance = en_sorted_coefs)

  # Filter rows where Importance is non-zero
  en_non_zero_coefs <- en_sorted_coefs_names[en_sorted_coefs_names$Importance != 0, ]
  en_non_zero_coefs_all[[i]] <- en_non_zero_coefs

  # Extract Feature names
  en_imp_features <- en_non_zero_coefs$Feature



  # Random Forest

  # Extract feature importance
  rf_importance_df <- data.frame(Feature = rownames(rf_outer_model$importance),
                                 Importance = rf_outer_model$importance[, "MeanDecreaseAccuracy"])

  rf_importance_df_reordered <- rf_importance_df[order(rf_importance_df$Importance, decreasing = TRUE), ]

  # Filter rows where Importance is not-negative
  rf_non_zero <- rf_importance_df_reordered[rf_importance_df_reordered$Importance > 0, ]
  rf_nonzero_importance[[i]] <- rf_non_zero

  rf_imp_features <- rf_non_zero$Feature



  # Classification Tree

  ct_imp <- ct_outer_model[["variable.importance"]]
  ct_imp <- data.frame(Feature = names(ct_imp))
  ct_imp_df[[i]] <- ct_imp

  ct_imp_features <-  ct_imp$Feature


  # adaBoost

  frame <- ada_outer_model[["trees"]][[1]][["frame"]]
  split_nodes <- frame[frame$var != "<leaf>", ]

  # Check if there are any split nodes before aggregating
  if (nrow(split_nodes) > 0) {
    # Summarize total impurity reduction per feature
    ada_feature_importance <- aggregate(dev ~ var, data = split_nodes, sum)

    # Sort by importance
    ada_feature_importance <- ada_feature_importance[order(-ada_feature_importance$dev), ]
    ada_feature_importance_sorted[[i]] <- ada_feature_importance

    ada_imp_features <- ada_feature_importance$var
  } else {
    # No splits; assign NA
    ada_feature_importance_sorted[[i]] <- NA
    ada_imp_features <- NA
  }


  # XGBoost
  xgb_sorted_imp <- xgb.importance(model = xgb_outer_model) # if this model in a round doesn't work, this will be an empty data table
  xgb_imp[[i]] <- xgb_sorted_imp

  xgb_imp_features <- xgb_sorted_imp$Feature



  # Feature Rankings
  # The following is a heatmap indicating the number of times each feature ranked 1st through `r length(genes)` number of features.

  # Helper to get safe length even if object is NA
  safe_length <- function(x) {
    if (is.null(x) || all(is.na(x))) {
      return(0)
    } else {
      return(length(x))
    }
  }

  # Compute the max length safely
  max_features <- max(
    safe_length(nb_imp_features),
    safe_length(log_imp_features),
    safe_length(lasso_imp_features),
    safe_length(en_imp_features),
    safe_length(rf_imp_features),
    safe_length(ct_imp_features),
    safe_length(ada_imp_features),
    safe_length(xgb_imp_features)
  )

  # Updated padding function to handle NA
  pad_features <- function(features, max_length) {
    if (is.null(features) || all(is.na(features))) {
      return(rep(NA, max_length))
    } else {
      length(features) <- max_length
      return(features)
    }
  }

  # Apply the padding function to all feature lists
  nb_imp_features <- pad_features(nb_imp_features, max_features)
  log_imp_features <- pad_features(log_imp_features, max_features)
  lasso_imp_features <- pad_features(lasso_imp_features, max_features)
  en_imp_features <- pad_features(en_imp_features, max_features)
  rf_imp_features <- pad_features(rf_imp_features, max_features)
  ct_imp_features <- pad_features(ct_imp_features, max_features)
  ada_imp_features <- pad_features(ada_imp_features, max_features)
  xgb_imp_features <- pad_features(xgb_imp_features, max_features)

  # Combine into data frame
  combined_features <- data.frame(
    NB = nb_imp_features,
    LR = log_imp_features,
    LASSO = lasso_imp_features,
    EN = en_imp_features,
    RF = rf_imp_features,
    CT = ct_imp_features,
    ADA = ada_imp_features,
    XGB = xgb_imp_features,
    stringsAsFactors = FALSE
  )


  combined_features_all[[i]] <- combined_features

  # Get the ranks for each feature
  ranked_combined_features <- apply(combined_features, 2, function(x) rank(x, ties.method = "first"))

  # Extract all feature names from any column and sort them alphabetically
  sorted_feature_names <- sort(combined_features[, 1])

  # report the number of times a character in sorted_feature_names appears in a row in combined_features
  # Create a new matrix to store the counts
  count_matrix <- matrix(0, nrow = nrow(combined_features), ncol = length(sorted_feature_names))

  # Set row names to match the rows of combined_features
  rownames(count_matrix) <- rownames(combined_features)

  # Set column names to be the features in sorted_feature_names
  colnames(count_matrix) <- sorted_feature_names

  # Loop through each feature and count occurrences in each row, skipping NAs
  for (k in seq_along(sorted_feature_names)) {
    feature <- sorted_feature_names[k]
    count_matrix[, k] <- apply(combined_features, 1, function(row) {
      # Count occurrences, ignoring NA
      sum(!is.na(row) & row == feature)
    })
  }

  # Convert the count_matrix to a dataframe for easier visualization
  imp_feature_count_df[[i]] <- as.data.frame(count_matrix)


  setTxtProgressBar(pbo,i)

} # End of Outer Loop
close(pbo)

