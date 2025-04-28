# Define the models in the same order as your original color usage
model_order <- c("Naive.Bayes", "Logistic", "LASSO", "EN", "RF", "Class.Tree", "adaBoost", "XGBoost")

# Assign colors from the Paired palette in the correct order
colors <- brewer.pal(8, "Paired")
names(colors) <- model_order

# Filter and reorder AUC summary data
auc_summary <- summary_table %>%
  filter(Metric == "AUC") %>%
  mutate(Model = factor(Model, levels = model_order))

# Plot with consistent colors
ggplot(auc_summary, aes(x = Model, y = Mean, fill = Model)) +
  geom_bar(stat = "identity", color = "black", width = 0.7) +
  geom_errorbar(aes(ymin = Mean - SEM, ymax = Mean + SEM), width = 0.2) +
  scale_fill_manual(values = colors) +
  theme_minimal(base_size = 14) +
  labs(
    title = "AUC Comparison Across ML Models",
    x = "Model",
    y = "Mean AUC ± SEM"
  ) +
  ylim(0, 1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")
