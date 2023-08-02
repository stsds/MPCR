library(Matrix)
library(rbenchmark)
library(MMPR)


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


run_chol_benchmark <- function(n, tile_size, replication, times, num_threads) {
  matrix <- generate_postive_matrix_alt(n)
  num_tiles <- (n^2) / (tile_size^2)
  tiles_per_row <- n / tile_size

  precision_single <- matrix(rep("single", times = num_tiles), nrow = tiles_per_row)
  precision_double <- matrix(rep("double", times = num_tiles), nrow = tiles_per_row)
  diag(precision_single) <- "double"


  mmpr_tile_matrix_single <- new(MMPRTile, n, n, tile_size, tile_size, matrix, precision_single)
  mmpr_tile_matrix_double <- new(MMPRTile, n, n, tile_size, tile_size, matrix, precision_double)


  cat("\n\n\n")
  cat("Running tile chol benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  chol(mmpr_tile_matrix_single, overwrite_input = FALSE, num_threads = num_threads),
                  chol(mmpr_tile_matrix_double, overwrite_input = FALSE, num_threads = num_threads),
                  columns = c("test", "replications", "elapsed")))

  cat("\n\n\n")
  cat("Running tile chol benchmark with input overwrite \n")
  print(benchmark(replications = rep(1, 1),
                  chol(mmpr_tile_matrix_single, overwrite_input = TRUE, num_threads = num_threads),
                  chol(mmpr_tile_matrix_double, overwrite_input = TRUE, num_threads = num_threads),
                  columns = c("test", "replications", "elapsed")))


}

# Define the arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 5) {
  cat("\n\n\n\n")
  stop("Please provide correct arguments, 1-matrix_size 2-number_of_replication 3-times")
}

mat_size <- as.integer(args[1])
tile_size <- as.integer(args[2])
replication <- as.integer(args[3])
times <- as.integer(args[4])
threads <- as.integer(args[5])

cat("Matrix size : ")
cat(paste(mat_size, mat_size, sep = "*"))
cat("\n")

cat("Tile size : ")
cat(paste(tile_size, tile_size, sep = "*"))
cat("\n")

cat("replication : ")
cat(replication)
cat("times : ")
cat(times)

cat("\n")
cat("threads : ")
cat(threads)
cat("\n")


run_chol_benchmark(mat_size, tile_size, replication, times, threads)






