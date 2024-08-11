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


run_eigen_becnhmark <- function(n, replication, times,operation_placement) {
  cat("\n\n\n")

  cat("Matrix : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  if (n > 20000) {
    matrix <- generate_matrix_big(n, n)
  }
  else {
    matrix <- matrix(rnorm(n^2), ncol = n)
  }


  MPCR_single <- as.MPCR(matrix, n, n, "single",operation_placement)
  MPCR.SetOperationPlacement(operation_placement)

  cat("\n")
  print(benchmark(replications = rep(replication, times),
                  eigen(MPCR_single),
                  columns = c("test", "replications", "elapsed")))

  MPCR_single$FreeGPU()
  MPCR_single$FreeCPU()

  MPCR_double <- as.MPCR(matrix, n, n, "double",operation_placement)
  MPCR.SetOperationPlacement(operation_placement)

  cat("\n")
  print(benchmark(replications = rep(replication, times),
                  eigen(MPCR_double),
                  columns = c("test", "replications", "elapsed")))

  cat("\n")
}

# Define the arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 4) {
  cat("\n\n\n\n")
  stop("Please provide correct arguments, 1-matrix_size 2-number_of_replication 3-times 4-operation_placement")
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

run_eigen_becnhmark(mat_size, replication, times,operation_placement)