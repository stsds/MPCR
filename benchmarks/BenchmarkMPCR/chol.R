library(Matrix)
library(rbenchmark)
library(MPCR)


generate_positive_matrix <- function(M, nu = 1, beta = 0.1, sigma_sq = 1) {

  locs <- cbind(rep(0:(M - 1), M) / (M - 1), rep(0:(M - 1), each = M) / (M - 1))
  x <- as.matrix(dist(locs)) # distance matrix

  if (nu == 0.5)
    return(sigma_sq * exp(-x / beta))
  ismatrix <- is.matrix(x)
  if (ismatrix) { nr = nrow(x); nc = ncol(x) }
  x <- c(x / beta)
  output <- rep(1, length(x))
  n <- sum(x > 0)
  if (n > 0) {
    x1 <- x[x > 0]
    output[x > 0] <-
      (1 / ((2^(nu - 1)) * gamma(nu))) *
        (x1^nu) *
        besselK(x1, nu)
  }
  if (ismatrix) {
    output <- matrix(output, nr, nc)
  }
  return(sigma_sq * output)

}

generate_postive_matrix_alt <- function(n) {
  A <- matrix(rnorm(n^2), ncol = n)
  A <- A %*% t(A) + n * diag(n)

  return(A)
}


run_chol_benchmark <- function(n, replication, times,operation_placement) {
  matrix <- generate_postive_matrix_alt(n)


  MPCR_matrix_single <- as.MPCR(matrix, n, n, "single",operation_placement)
  MPCR_matrix_double <- as.MPCR(matrix, n, n, "double",operation_placement)

  MPCR.SetOperationPLacement(operation_placement)

  cat("\n\n\n")
  cat("Running chol benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  chol(MPCR_matrix_single),
                  chol(MPCR_matrix_double),
                  columns = c("test", "replications", "elapsed")))


  MPCR_chol_single <- chol(MPCR_matrix_single)
  MPCR_chol_double <- chol(MPCR_matrix_double)

  cat("Running chol2inv benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  chol2inv(MPCR_chol_single),
                  chol2inv(MPCR_chol_double),
                  columns = c("test", "replications", "elapsed")))

}

# Define the arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 4) {
  cat("\n\n\n\n")
  stop("Please provide correct arguments, 1-matrix_size 2-number_of_replication 3-times")
}

mat_size <- as.integer(args[1])
replication <- as.integer(args[2])
times <- as.integer(args[3])
operation_placement <- toString(args[4])

cat("Matrix size : ")
cat(paste(mat_size, mat_size, sep = "*"))
cat("\n")
cat("replication : ")
cat(replication)
cat("times : ")
cat(times)
cat("\n")
cat("Operation Placement : ")
cat(operation_placement)
cat("\n")

run_chol_benchmark(mat_size, replication, times,operation_placement)






