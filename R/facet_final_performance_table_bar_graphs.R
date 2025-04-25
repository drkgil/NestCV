# Define the correct model order
model_order <- c("Naive.Bayes", "Logistic", "LASSO", "EN", "RF", "Class.Tree", "adaBoost", "XGBoost")
colors <- brewer.pal(8, "Paired")
names(colors) <- model_order

# Apply filtering and factor level ordering
filtered_summary <- summary_table %>%
  filter(!Metric %in% c("TP", "TN")) %>% # include any metrics here you DON'T want to include
  mutate(Model = factor(Model, levels = model_order))

# Faceted plot with correct order and error bars
ggplot(filtered_summary, aes(x = Model, y = Mean, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_errorbar(aes(ymin = Mean - SEM, ymax = Mean + SEM),
                width = 0.2, position = position_dodge(width = 0.9)) +
  scale_fill_manual(values = colors) +
  facet_wrap(~ Metric, scales = "free_y") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  ) +
  labs(
    title = "Performance Metrics by Model",
    x = "Model",
    y = "Mean"
  )
