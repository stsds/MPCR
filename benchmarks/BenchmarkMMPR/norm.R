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


run_norm_benchmark <- function(n, replication, times) {

  if (n > 20000) {
    A <- generate_matrix_big(n, n)
  }
  else {
    A <- matrix(rnorm(n^2), ncol = n)
  }


  mmpr_single <- as.MMPR(A, n, n, "single")
  mmpr_double <- as.MMPR(A, n, n, "double")

  cat("Matrix : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  cat("\n")
  print(benchmark(replications = rep(replication, times),
                  norm(mmpr_single, "O"),
                  norm(mmpr_single, "I"),
                  norm(mmpr_single, "F"),
                  norm(mmpr_single, "M"),
                  norm(mmpr_double, "O"),
                  norm(mmpr_double, "I"),
                  norm(mmpr_double, "F"),
                  norm(mmpr_double, "M"),
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

run_norm_benchmark(mat_size, replication, times)