library(rbenchmark)
library(MPCR)



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

run_rcond_benchmark <- function(m, n, replication, times) {

  # Create a random matrix of size n x n
  if (n > 20000) {
    A <- generate_matrix_big(n, n)
    # Create a random matrix of size n x n
    B <- generate_matrix_big(m,n)
  }
  else {
    A <- matrix(rnorm(n^2), ncol = n)
    # Create a random matrix of size n x n
    B <- matrix(rnorm(n * m), ncol = n)
  }




  cat("Matrix A : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  cat("Matrix B : ")
  cat(paste(m, n, sep = "*"))
  cat("\n\n")


  MPCR_single_a <- as.MPCR(A, n, n, "single")
  MPCR_double_a <- as.MPCR(A, n, n, "double")


  MPCR_single_b <- as.MPCR(B, m, n, "single")
  MPCR_double_b <- as.MPCR(B, m, n, "double")

  cat("Running rcond bencmark")
  print(benchmark(replications = rep(replication, times),
                  rcond(MPCR_single_a),
                  rcond(MPCR_double_a),
                  rcond(MPCR_single_b),
                  rcond(MPCR_double_b),
                  columns = c("test", "replications", "elapsed")))

}

# Define the arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 4) {
  cat("\n\n\n\n")
  stop("Please provide correct arguments, 1-row 2-col 3-number_of_replication 4-times")
}

row <- as.integer(args[1])
col <- as.integer(args[2])
replication <- as.integer(args[3])
times <- as.integer(args[4])

cat("Matrix size : ")
cat(paste(row, col, sep = "*"))
cat("\n")
cat("replication : ")
cat(replication)
cat("times : ")
cat(times)

run_rcond_benchmark(row, col, replication, times)