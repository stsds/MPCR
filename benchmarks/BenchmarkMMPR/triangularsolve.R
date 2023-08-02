library(rbenchmark)
library(MMPR)


generate_matrix_big <- function(n, m) {
  # Set the matrix dimensions
  nrows <- n
  ncols <- m

  # Set the number of submatrices
  n_submatrices <- 4

  # Determine the number of rows and columns for each submatrix
  sub_nrows <- ceiling(nrows / sqrt(n_submatrices))
  sub_ncols <- ceiling(ncols / sqrt(n_submatrices))

  # Generate random values for each submatrix
  sub_matrices <- lapply(1:n_submatrices, function(x) {
    rand_vals <- rnorm(sub_nrows * sub_ncols)
    matrix(rand_vals, nrow = sub_nrows, ncol = sub_ncols)
  })

  # Combine the submatrices into a single matrix
  my_matrix <- do.call(cbind, lapply(split(sub_matrices, rep(1:2, each = 2)), function(x) {
    do.call(rbind, x)
  }))

  return(my_matrix)
}

run_backsolve_benchmark <- function(n, replication, times) {
  cat("\n\n")
  cat("Matrix  : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  if (n > 20000) {
    U <- generate_matrix_big(n, n)
  }
  else {
    U <- matrix(rnorm(n^2), ncol = n)
  }

  U <- upper.tri(U)
  diag(U) <- runif(n, 0.1, 1)

  # Create a random right-hand side vector of length n

  b <- matrix(rnorm(n^2), ncol = n)
  b <- upper.tri(b)
  diag(b) <- runif(n, 0.1, 1)

  mmpr_single_U <- as.MMPR(U, n, n, "single")
  mmpr_double_U <- as.MMPR(U, n, n, "double")


  mmpr_single_b <- as.MMPR(b, n, n, precision = "single")
  mmpr_double_b <- as.MMPR(b, n, n, precision = "double")


  cat("\n\n")
  cat("Running backsolve benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  backsolve(mmpr_single_U, mmpr_single_b),
                  backsolve(mmpr_double_U, mmpr_double_b),
                  columns = c("test", "replications", "elapsed")))


}


run_forwardsolve_benchmark <- function(n, replication, times) {
  cat("\n\n")
  cat("Matrix  : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  if (n > 20000) {
    L <- generate_matrix_big(n, n)
  }
  else {
    L <- matrix(rnorm(n^2), ncol = n)
  }
  L <- lower.tri(L)
  diag(L) <- runif(n, 0.1, 1)

  # Create a random right-hand side vector of length n
  b <- matrix(rnorm(n^2), ncol = n)
  b <- upper.tri(b)
  diag(b) <- runif(n, 0.1, 1)


  mmpr_single_L <- as.MMPR(L, n, n, "single")
  mmpr_double_L <- as.MMPR(L, n, n, "double")

  mmpr_single_b <- as.MMPR(b, n, n, precision = "single")
  mmpr_double_b <- as.MMPR(b, n, n, precision = "double")

  cat("\n\n")
  cat("Running forwardsolve benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  forwardsolve(mmpr_single_L, mmpr_single_b),
                  forwardsolve(mmpr_double_L, mmpr_double_b),
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


run_backsolve_benchmark(mat_size, replication, times)
run_forwardsolve_benchmark(mat_size, replication, times)

