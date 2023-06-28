library(rbenchmark)

run_solve_benchmark <- function(n, replication, times) {

  # Create a random matrix
  set.seed(123)
  A <- matrix(rnorm(n * n), nrow = n, ncol = n)

  # Create a random vector
  b <- rnorm(n)


  cat("Matrix A : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  cat("Matrix B : ")
  cat(n)
  cat("\n")

  cat("\n\n")
  cat("Running solve benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  solve(A, b),
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


run_solve_benchmark(mat_size, replication, times)


