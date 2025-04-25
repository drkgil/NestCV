# Convert results to a dataframe
outer_results_df <- do.call(rbind, lapply(outer_results, as.data.frame))

# Save the predictions into a consolidated format
consolidated_outer_results <- lapply(seq_along(outer_results[[1]]), function(i) {
  do.call(c, lapply(outer_results, `[[`, i))
})
names(consolidated_outer_results) <- c("actual","nb", "log", "lasso", "en", "rf", "classtree", "adaboost", "xgboost")

actual_outcomes <- consolidated_outer_results$actual
naive_bayes_predictions <- consolidated_outer_results$nb
log_predictions <- consolidated_outer_results$log
lasso_predictions <- consolidated_outer_results$lasso
en_predictions <- consolidated_outer_results$en
rf_predictions <- consolidated_outer_results$rf
ct_predictions <- consolidated_outer_results$classtree
adaboost_predictions <- consolidated_outer_results$adaboost
xgboost_predictions <- consolidated_outer_results$xgboost



# Merge all iterations into a single table
final_performance_table <- do.call(rbind, all_metrics)

# Reshape from wide to long format
long_df <- final_performance_table %>%
  pivot_longer(-Metric, names_to = "Model", values_to = "Value") %>%
  mutate(Value = as.numeric(Value))  # Ensure numeric values for aggregation

# Calculate mean and SEM per metric per model
summary_table <- long_df %>%
  group_by(Metric, Model) %>%
  summarise(
    Mean = mean(Value, na.rm = TRUE),
    SEM = sd(Value, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Optionally, format Mean ± SEM as a string
summary_table <- summary_table %>%
  mutate(`Mean ± SEM` = sprintf("%.3f ± %.3f", Mean, SEM))

# Optional: Make it a pretty table
library(knitr)
library(kableExtra)

summary_table %>%
  select(Metric, Model, `Mean ± SEM`) %>%
  pivot_wider(names_from = Model, values_from = `Mean ± SEM`) %>%
  kable("html", caption = "Performance Metrics Summary (Mean ± SEM)") %>%
  kable_styling(full_width = FALSE)
