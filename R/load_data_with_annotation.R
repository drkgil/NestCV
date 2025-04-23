#' Load and align data with annotations
#'
#' This function loads a data matrix and associated annotations from either files or data frames,
#' and returns them in a standardized format.
#'
#' @param data Either a data frame or a file path to a CSV/XLSX file containing subjects in rows and features in columns.
#' @param annotation Either a factor/vector, a data frame, or a file path to a CSV/XLSX.
#' @param id_col Optional name of column to use as rownames for both data and annotation. Default is NULL.
#' @param annotation_col Optional name of column in annotation data frame to use as the classification factor if there are multiple columns.
#'
#' @return A list with:
#' \describe{
#'   \item{data}{A data frame of the input data}
#'   \item{annotation}{A factor vector aligned to the data by rowname}
#' }
#' @export
load_data_with_annotation <- function(data, annotation, id_col = NULL, annotation_col = NULL) {
  # Helper: read data
  read_input <- function(input, use_first_col_as_rownames = FALSE) {
    if (is.character(input)) {
      if (grepl("\\.csv$", input)) {
        df <- read.csv(input, stringsAsFactors = FALSE, row.names = if (use_first_col_as_rownames) 1 else NULL)
      } else if (grepl("\\.xlsx$", input)) {
        if (!requireNamespace("readxl", quietly = TRUE)) {
          stop("Package 'readxl' is required to read Excel files.")
        }
        df <- readxl::read_excel(input)
        df <- as.data.frame(df)
      } else {
        stop("Unsupported file type. Please use .csv or .xlsx.")
      }
      return(df)
    } else if (is.data.frame(input)) {
      return(input)
    } else {
      stop("Input must be a file path or a data frame.")
    }
  }


  # Read and prep data
  df <- read_input(data, use_first_col_as_rownames = is.null(id_col))
  if (!is.null(id_col)) {
    rownames(df) <- df[[id_col]]
    df[[id_col]] <- NULL
  }

  # Read and prep annotation
  ann_df <- read_input(annotation, use_first_col_as_rownames = is.null(id_col))
  if (!is.null(id_col) && is.data.frame(ann_df)) {
    rownames(ann_df) <- ann_df[[id_col]]
    ann_df[[id_col]] <- NULL
  }


  # Extract annotation factor
  if (is.data.frame(ann_df)) {
    if (!is.null(annotation_col)) {
      if (!(annotation_col %in% colnames(ann_df))) {
        stop("annotation_col not found in annotation data.")
      }
      annotation_factor <- as.factor(ann_df[[annotation_col]])
      names(annotation_factor) <- rownames(ann_df)
    } else if (ncol(ann_df) == 1) {
      annotation_factor <- as.factor(ann_df[[1]])
      names(annotation_factor) <- rownames(ann_df)
    } else {
      stop("Annotation data has multiple columns. Please specify 'annotation_col'.")
    }
  } else if (is.vector(ann_df) || is.factor(ann_df)) {
    annotation_factor <- as.factor(ann_df)
  } else {
    stop("Unsupported annotation input.")
  }

  # Subset to matched rownames
  matched_ids <- intersect(rownames(df), names(annotation_factor))
  if (length(matched_ids) == 0) {
    stop("No overlapping row names between data and annotation.")
  }
  if (length(matched_ids) < length(rownames(df)) || length(matched_ids) < length(annotation_factor)) {
    warning("Some rows in data or annotation were dropped due to mismatched rownames.")
  }


  df <- df[matched_ids, , drop = FALSE]
  annotation_factor <- annotation_factor[matched_ids]

  return(list(
    data = df,
    annotation = annotation_factor
  ))
}
