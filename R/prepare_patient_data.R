#' Prepare filtered data and outcome for modeling
#'
#' @param data_obj Data frame or matrix containing features of interest
#' @param gene_list Character vector of gene names to keep (or features)
#' @param annotation Column name in annotation for outcome (must be binary/categorical)
#' @param treatment_col Optional column name in annotation to filter by treatment group
#' @param treatment_value Optional value in treatment_col to keep
#'
#' @return A list with `X` (features) and `Y` (binary outcome)
#' @export
prepare_patient_data <- function(data, annotation, gene_list) {
  # Ensure data is a data frame
  data <- as.data.frame(data)
  # "sanitize" column names e.g., remove spaces, special characters, or make them syntactically valid in R
  colnames(data) <- make.names(colnames(data))

  # If gene_list is not provided, use all columns
  if (is.null(gene_list)) {
    gene_list <- colnames(data)
    message("No gene list provided; using all columns from data.")
  }

  # Subset data to include only selected genes
  patientFeaturesData <- data[, gene_list, drop = FALSE]

  # Ensure annotation is a data frame
  annotation <- as.data.frame(annotation)
  # "sanitize" column name
  colnames(annotation) <- make.names(colnames(annotation))
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
