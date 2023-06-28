library(rbenchmark)
library(MMPR)


run_svd_benchmark <- function(n, replication, times) {

  cat("Matrix A : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  # Create a random matrix of size n x n
  A <- matrix(rnorm(n^2), ncol = n)
  mmpr_single <- as.MMPR(A, n, n, "single")
  mmpr_double <- as.MMPR(A, n, n, "double")


  cat("\n\n")
  cat("Running svd benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  svd(mmpr_single),
                  svd(mmpr_double),
                  columns = c("test", "replications", "elapsed")))

  cat("\n\n")
  cat("Running La.svd benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  La.svd(mmpr_single),
                  La.svd(mmpr_double),
                  columns = c("test", "replications", "elapsed")))


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


run_svd_benchmark(mat_size, replication, times)


