#' @title Load required packages
#' @description Loads packages listed in the `cran_packages` object.
#' @export
load_packages <- function() {
  for (pkg in cran_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
    library(pkg, character.only = TRUE)
  }
}

