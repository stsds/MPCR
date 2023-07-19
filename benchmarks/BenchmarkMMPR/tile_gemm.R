library(Matrix)
library(rbenchmark)
library(MMPR)


run_gemm_benchmark <- function(row, col, tile_row, tile_col, replication, times, threads) {
  cat("\n\n\n")

  cat("Matrix 1 : ")
  cat(paste(row, col, sep = "*"))
  cat("\n")

  cat("Matrix 2 : ")
  cat(paste(col, row, sep = "*"))
  cat("\n")
  matrix_1 <- matrix(runif(row * col, min = 0.5, max = 2), nrow = row, ncol = col)
  matrix_2 <- matrix(runif(row * col, min = -3, max = 2), nrow = col, ncol = row)
  print(nrow(matrix_1))
  print(ncol(matrix_1))
  print(nrow(matrix_2))
  print(ncol(matrix_2))

  num_tiles <- (row * col) / (tile_row * tile_col)
  tiles_per_col <- row / tile_row
  zeros <- replicate(row * col, 0)

  precision_single <- matrix(rep("single", times = num_tiles), nrow = tiles_per_col)
  precision_double <- matrix(rep("double", times = num_tiles), nrow = tiles_per_col)

  mmpr_tile_matrix_single_1 <- new(MMPRTile, row, col, tile_row, tile_col, matrix_1, precision_single)
  mmpr_tile_matrix_single_2 <- new(MMPRTile, col, row, tile_col, tile_row, matrix_2, precision_single)
  mmpr_tile_matrix_single_3 <- new(MMPRTile, col, col, tile_col, tile_col, zeros, precision_single)

  mmpr_tile_matrix_double_1 <- new(MMPRTile, row, col, tile_row, tile_col, matrix_1, precision_double)
  mmpr_tile_matrix_double_2 <- new(MMPRTile, col, row, tile_col, tile_row, matrix_2, precision_double)
  mmpr_tile_matrix_double_3 <- new(MMPRTile, col, col, tile_col, tile_col, zeros, precision_double)

  print(benchmark(replications = rep(replication, times),
                  MMPRTile.gemm(a = mmpr_tile_matrix_single_1, b = mmpr_tile_matrix_single_2, c = mmpr_tile_matrix_single_3, transpose_a = FALSE, transpose_b = FALSE, alpha = 1, beta = 1, num_thread = threads),
                  columns = c("test", "replications", "elapsed")))


  print(benchmark(replications = rep(replication, times),
                  MMPRTile.gemm(a = mmpr_tile_matrix_double_1, b = mmpr_tile_matrix_double_2, c = mmpr_tile_matrix_double_3, transpose_a = FALSE, transpose_b = FALSE, alpha = 1, beta = 1, num_thread = threads),
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






