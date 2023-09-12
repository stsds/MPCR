library(Matrix)
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


run_gemm_benchmark <- function(row, col, tile_row, tile_col, replication, times, threads) {
  cat("\n\n\n")

  cat("Matrix 1 : ")
  cat(paste(row, col, sep = "*"))
  cat("\n")

  cat("Matrix 2 : ")
  cat(paste(col, row, sep = "*"))
  cat("\n")

  matrix_1 <- generate_matrix_big(row, col)
  matrix_2 <- generate_matrix_big(col, row)


  print(nrow(matrix_1))
  print(ncol(matrix_1))
  print(nrow(matrix_2))
  print(ncol(matrix_2))

  num_tiles <- (row / tile_row) * (col / tile_col)

  tiles_per_col <- row / tile_row
  zeros <- generate_matrix_big(row, col)

  precision_single <- matrix(rep("single", times = num_tiles), nrow = tiles_per_col)
  precision_double <- matrix(rep("double", times = num_tiles), nrow = tiles_per_col)

  MPCR_tile_matrix_single_1 <- new(MPCRTile, row, col, tile_row, tile_col, matrix_1, precision_single)
  MPCR_tile_matrix_single_2 <- new(MPCRTile, col, row, tile_col, tile_row, matrix_2, precision_single)
  MPCR_tile_matrix_single_3 <- new(MPCRTile, col, col, tile_col, tile_col, zeros, precision_single)

  MPCR_tile_matrix_double_1 <- new(MPCRTile, row, col, tile_row, tile_col, matrix_1, precision_double)
  MPCR_tile_matrix_double_2 <- new(MPCRTile, col, row, tile_col, tile_row, matrix_2, precision_double)
  MPCR_tile_matrix_double_3 <- new(MPCRTile, col, col, tile_col, tile_col, zeros, precision_double)

  print(benchmark(replications = rep(replication, times),
                  MPCRTile.gemm(a = MPCR_tile_matrix_single_1, b = MPCR_tile_matrix_single_2, c = MPCR_tile_matrix_single_3, transpose_a = FALSE, transpose_b = FALSE, alpha = 1, beta = 1, num_threads = threads),
                  columns = c("test", "replications", "elapsed")))


  print(benchmark(replications = rep(replication, times),
                  MPCRTile.gemm(a = MPCR_tile_matrix_double_1, b = MPCR_tile_matrix_double_2, c = MPCR_tile_matrix_double_3, transpose_a = FALSE, transpose_b = FALSE, alpha = 1, beta = 1, num_threads = threads),
                  columns = c("test", "replications", "elapsed")))

  cat("\n")
}

# Define the arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 7) {
  cat("\n\n\n\n")
  stop("Please provide correct arguments, 1-matrix_size 2-number_of_replication 3-times")
}

row <- as.integer(args[1])
col <- as.integer(args[2])
tile_row <- as.integer(args[3])
tile_col <- as.integer(args[4])
replication <- as.integer(args[5])
times <- as.integer(args[6])
threads <- as.integer(args[7])

cat("Matrix size : ")
cat(paste(row, col, sep = "*"))
cat("\n")

cat("Tile size : ")
cat(paste(tile_row, tile_col, sep = "*"))
cat("\n")

cat("replication : ")
cat(replication)
cat("times : ")
cat(times)

cat("\n")
cat("threads : ")
cat(threads)
cat("\n")


run_gemm_benchmark(row, col, tile_row, tile_col, replication, times, threads)






