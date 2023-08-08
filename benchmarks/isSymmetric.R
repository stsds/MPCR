library(rbenchmark)
library(MMPR)


run_issymmeric_benchmark <- function(n, replication, times) {


  set.seed(123)
  matrix_symmetric <- matrix(rnorm(n * n), nrow = n, ncol = n)
  matrix_symmetric <- upper.tri(matrix_symmetric) + t(upper.tri(matrix_symmetric, diag = TRUE))


  matrix_sq_nonsymmetric <- matrix(rnorm(n * n), nrow = n, ncol = n)
  matrix_nonsymmetric <- matrix(rnorm(n * (n / 2)), nrow = n, ncol = n / 2)


  mmpr_matrix_symmetric_double <- as.MMPR(matrix_symmetric, n, n, "double")
  mmpr_matrix_sq_nonsymmetric_double <- as.MMPR(matrix_sq_nonsymmetric, n, n, "double")
  mmpr_matrix_nonsymmetric_double <- as.MMPR(matrix_nonsymmetric, n, n, "double")

  mmpr_matrix_symmetric_single <- as.MMPR(matrix_symmetric, n, n, "single")
  mmpr_matrix_sq_nonsymmetric_single <- as.MMPR(matrix_sq_nonsymmetric, n, n, "single")
  mmpr_matrix_nonsymmetric_single <- as.MMPR(matrix_nonsymmetric, n, n, "single")


  cat("Matrix  : ")
  cat(paste(n, n, sep = "*"))

  cat("Matrix non square : ")
  cat(paste(n, (n / 2), sep = "*"))

  cat("\n")
  cat("\n\n\n")
  cat("Running isSymmetric on symmetric matrix \n")

  print(benchmark(replications = rep(replication, times),
                  isSymmetric(matrix_symmetric),
                  isSymmetric(mmpr_matrix_symmetric_single),
                  isSymmetric(mmpr_matrix_symmetric_double),
                  columns = c("test", "replications", "elapsed")))

  cat("\n\n\n")
  cat("Running isSymmetric on non-symmetric square matrix \n")

  print(benchmark(replications = rep(replication, times),
                  isSymmetric(matrix_sq_nonsymmetric),
                  isSymmetric(mmpr_matrix_sq_nonsymmetric_single),
                  isSymmetric(mmpr_matrix_sq_nonsymmetric_double),
                  columns = c("test", "replications", "elapsed")))

  cat("\n\n\n")
  cat("Running isSymmetric on non-symmetric matrix \n")

  print(benchmark(replications = rep(replication, times),
                  isSymmetric(matrix_nonsymmetric),
                  isSymmetric(mmpr_matrix_nonsymmetric_single),
                  isSymmetric(mmpr_matrix_nonsymmetric_double),
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


run_issymmeric_benchmark(mat_size, replication, times)