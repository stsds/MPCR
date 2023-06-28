library(rbenchmark)
library(MMPR)


run_rcond_benchmark <- function(m, n, replication, times) {

  # Create a random matrix of size n x n
  A <- matrix(rnorm(n^2), ncol = n)

  # Create a random matrix of size n x n
  B <- matrix(rnorm(n * m), ncol = n)

  cat("Matrix A : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  cat("Matrix B : ")
  cat(paste(m, n, sep = "*"))
  cat("\n\n")


  mmpr_single_a <- as.MMPR(A, n, n, "single")
  mmpr_double_a <- as.MMPR(A, n, n, "double")


  mmpr_single_b <- as.MMPR(B, m, n, "single")
  mmpr_double_b <- as.MMPR(B, m, n, "double")

  cat("Running rcond bencmark")
  print(benchmark(replications = rep(replication, times),
                  rcond(mmpr_single_a),
                  rcond(mmpr_double_a),
                  rcond(mmpr_single_b),
                  rcond(mmpr_double_b),
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