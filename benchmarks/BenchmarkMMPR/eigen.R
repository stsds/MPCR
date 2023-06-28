library(rbenchmark)
library(MMPR)

run_eigen_becnhmark <- function(n, replication, times) {
  cat("\n\n\n")

  cat("Matrix : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  matrix <- matrix(rnorm(n^2), ncol = n)

  mmpr_single <- as.MMPR(matrix, n, n, "single")
  mmpr_double <- as.MMPR(matrix, n, n, "double")

  cat("\n")
  print(benchmark(replications = rep(replication, times),
                  eigen(mmpr_single),
                  eigen(mmpr_double),
                  columns = c("test", "replications", "elapsed")))

  cat("\n")
}

# Define the arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  cat("\n\n\n\n")
  stop("Please provide correct arguments, 1-matrix_size 2-number_of_replication 3-times")
}

mat_size <- as.integer(args[1])
replication <- as.integer(args[2])
times <- as.integer(args[3])

cat("Matrix size : ")
cat(paste(mat_size, mat_size, sep = "*"))
cat("\n")
cat("replication : ")
cat(replication)
cat("times : ")
cat(times)

run_eigen_becnhmark(mat_size, replication, times)