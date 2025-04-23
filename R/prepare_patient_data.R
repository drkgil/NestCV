#' Prepare filtered data and outcome for modeling
#'
#' @param data_obj List containing `data` and `annotation`
#' @param gene_list Character vector of gene names to keep
#' @param class_col Column name in annotation for outcome (must be binary/categorical)
#' @param treatment_col Optional column name in annotation to filter by treatment group
#' @param treatment_value Optional value in treatment_col to keep
#'
#' @return A list with `X` (features) and `Y` (binary outcome)
#' @export
prepare_patient_data <- function(data, annotation, gene_list) {
  # Subset data to include only selected genes
  patientFeaturesData <- data[, gene_list]

  # Add the annotation as a new column (assuming it's already in 0's and 1's)
  patientFeaturesData$Survival_death <- annotation

  # Prepare data matrices
  data_Y <- as.matrix(patientFeaturesData$Survival_death)
  data_X <- patientFeaturesData[, -ncol(patientFeaturesData)]  # Exclude the last column (Survival_death)

  # Assign to global environment
  assign("data_X", data_X, envir = .GlobalEnv)
  assign("data_Y", data_Y, envir = .GlobalEnv)
  assign("patientFeaturesData", patientFeaturesData, envir = .GlobalEnv)

  return(TRUE)  # Adding a return value for confirmation
}
