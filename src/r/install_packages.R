# The rocker/r-ver image points CRAN at a dated snapshot, so these
# installs resolve to fixed versions that match the R version.
packages <- c("readr", "dplyr")

install.packages(packages, repos = getOption("repos"))

for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) stop(paste("Failed to install", p))
  cat(sprintf("Installed %s version %s\n", p, as.character(packageVersion(p))))
}
