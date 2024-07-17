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


run_gemm_benchmark <- function(row, col, replication, times ,operation_placement) {
  cat("\n\n\n")

  cat("Matrix 1 : ")
  cat(paste(row, col, sep = "*"))
  cat("\n")

  cat("Matrix 2 : ")
  cat(paste(col, row, sep = "*"))
  cat("\n")
  matrix_1 <- generate_matrix_big(row, col)
  matrix_2 <- generate_matrix_big(row, col)

  print(nrow(matrix_1))
  print(ncol(matrix_1))
  print(nrow(matrix_2))
  print(ncol(matrix_2))

  MPCR_matrix_single_1 <- as.MPCR(matrix_1, row, col, "single",operation_placement)
  MPCR_matrix_double_1 <- as.MPCR(matrix_1, row, col, "double",operation_placement)


  MPCR_matrix_single_2 <- as.MPCR(matrix_2, col, row, "single",operation_placement)
  MPCR_matrix_double_2 <- as.MPCR(matrix_2, col, row, "double",operation_placement)

  MPCR.SetOperationPlacement(operation_placement)

  cat("\n\n")
  cat("Running crossprod benchmark with 2 input \n")
  print(benchmark(replications = rep(replication, times),
                  crossprod(MPCR_matrix_single_1, MPCR_matrix_single_2),
                  crossprod(MPCR_matrix_double_1, MPCR_matrix_double_2),
                  columns = c("test", "replications", "elapsed")))

  MPCR_matrix_single_3 <- as.MPCR(matrix_2, row, col, "single",operation_placement)
  MPCR_matrix_double_3 <- as.MPCR(matrix_2, row, col, "double",operation_placement)

  cat("\n\n")
  cat("Running tcrossprod benchmark with 2 input \n")
  print(benchmark(replications = rep(replication, times),
                  tcrossprod(MPCR_matrix_single_1, MPCR_matrix_single_3),
                  tcrossprod(MPCR_matrix_double_1, MPCR_matrix_double_3),
                  columns = c("test", "replications", "elapsed")))

  cat("\n\n\n")
  matrix_3 <- generate_matrix_big(row, row)


  MPCR_matrix_single_3 <- as.MPCR(matrix_3, row, row, "single",operation_placement)
  MPCR_matrix_double_3 <- as.MPCR(matrix_3, row, row, "double",operation_placement)
  cat("Matrix 3 : ")
  cat(paste(row, row, sep = "*"))
  cat("\n")
  cat("\n\n\n")
  cat("Running crossprod benchmark with 1 input \n")
  print(benchmark(replications = rep(replication, times),
                  crossprod(MPCR_matrix_single_3),
                  crossprod(MPCR_matrix_double_3),
                  columns = c("test", "replications", "elapsed")))


  cat("Running tcrossprod benchmark with 1 input \n")
  print(benchmark(replications = rep(replication, times),
                  tcrossprod(MPCR_matrix_single_3),
                  tcrossprod(MPCR_matrix_double_3),
                  columns = c("test", "replications", "elapsed")))


  cat("\n")
}

# Define the arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 5) {
  cat("\n\n\n\n")
  stop("Please provide correct arguments, 1-row 2-col 3-number_of_replication 4-times")
}

row <- as.integer(args[1])
col <- as.integer(args[2])
replication <- as.integer(args[3])
times <- as.integer(args[4])
operation_placement <- toString(args[5])


cat("Matrix size : ")
cat(paste(row, col, sep = "*"))
cat("\n")
cat("replication : ")
cat(replication)
cat("times : ")
cat(times)
cat("\n")
cat("Operation Placement : ")
cat(operation_placement)
cat("\n")

run_gemm_benchmark(row, col, replication, times , operation_placement)






