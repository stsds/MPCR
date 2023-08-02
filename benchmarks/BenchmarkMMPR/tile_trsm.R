library(rbenchmark)
library(MMPR)


run_trsm_benchmark <- function(n, tile_size, replication, times) {
  cat("\n\n")
  cat("Matrix  : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  num_tiles <- (n^2) / (tile_size^2)
  tiles_per_col <- n / tile_size


  precision_single <- matrix(rep("single", times = num_tiles), nrow = tiles_per_col)
  precision_double <- matrix(rep("double", times = num_tiles), nrow = tiles_per_col)


  U <- matrix(rnorm(n^2), ncol = n)
  U <- upper.tri(U)
  diag(U) <- runif(n, 0.1, 1)


  b <- matrix(rnorm(n^2), ncol = n)
  b <- upper.tri(b)
  diag(b) <- runif(n, 0.1, 1)
  # Create a random right-hand side vector of length n


  mmpr_tile_single_U <- new(MMPRTile, n, n, tile_size, tile_size, U, precision_single)
  mmpr_tile_double_U <- new(MMPRTile, n, n, tile_size, tile_size, U, precision_double)

  print("Here")
  mmpr_tile_single_b <- new(MMPRTile, n,n, tile_size,tile_size, b, precision_single)
  mmpr_tile_double_b <- new(MMPRTile, n,n, tile_size,tile_size, b, precision_double)
  print("Here")

  cat("\n\n")
  cat("Running tile trsm benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  MMPRTile.trsm(a = mmpr_tile_double_U, b = mmpr_tile_double_b, side = 'R', upper_triangle = TRUE, transpose = FALSE, alpha = 1),
                  MMPRTile.trsm(a = mmpr_tile_single_U, b = mmpr_tile_single_b, side = 'R', upper_triangle = TRUE, transpose = FALSE, alpha = 1),
                  columns = c("test", "replications", "elapsed")))


}


# Define the arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 4) {
  cat("\n\n\n\n")
  stop("Please provide correct arguments, 1-matrix_size 2-number_of_replication 3-times")
}

mat_size <- as.integer(args[1])
tile_size <- as.integer(args[2])
replication <- as.integer(args[3])
times <- as.integer(args[4])

cat("Matrix size : ")
cat(paste(mat_size, mat_size, sep = "*"))
cat("\n")

cat("Tile size A : ")
cat(paste(tile_size, tile_size, sep = "*"))
cat("\n")


cat("Tile size B : ")
cat(paste(1, tile_size, sep = "*"))
cat("\n")

cat("replication : ")
cat(replication)
cat("times : ")
cat(times)


run_trsm_benchmark(mat_size, tile_size, replication, times)






