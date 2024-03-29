library(rbenchmark)


run_backsolve_benchmark <- function(n, replication, times) {
  cat("\n\n")
  cat("Matrix  : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  U <- matrix(rnorm(n^2), ncol = n)
  U <- upper.tri(U)
  diag(U) <- runif(n, 0.1, 1)

  # Create a random right-hand side vector of length n
  b <- matrix(rnorm(n^2), ncol = n)
  b <- upper.tri(b)
  diag(b) <- runif(n, 0.1, 1)

  cat("\n\n")
  cat("Running backsolve benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  backsolve(U, b),
                  columns = c("test", "replications", "elapsed")))


}


run_forwardsolve_benchmark <- function(n, replication, times) {
  cat("\n\n")
  cat("Matrix  : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  L <- matrix(rnorm(n^2), ncol = n)
  L <- lower.tri(L)
  diag(L) <- runif(n, 0.1, 1)

  # Create a random right-hand side vector of length n
  b <- matrix(rnorm(n^2), ncol = n)
  b <- upper.tri(b)
  diag(b) <- runif(n, 0.1, 1)

  cat("\n\n")
  cat("Running backsolve benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  forwardsolve(L, b),
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
