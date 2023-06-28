library(rbenchmark)


run_qr_benchmark <- function(n, replication, times) {

  # Generate a random matrix of size n x n
  set.seed(123)
  matrix <- matrix(rnorm(n * n), nrow = n)

  cat("Matrix A : ")
  cat(paste(n, n, sep = "*"))

  cat("\n\n\n")
  cat("Running qr benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  qr(matrix, LAPACK = TRUE),
                  qr(matrix, LAPACK = FALSE),
                  columns = c("test", "replications", "elapsed")))

  qr_out <- qr(matrix, LAPACK = TRUE)


  cat("Running qr.R benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  qr.R(qr_out),
                  columns = c("test", "replications", "elapsed")))


  cat("Running qr.Q benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  qr.Q(qr_out),
                  columns = c("test", "replications", "elapsed")))


  b <- rnorm(n)
  cat("Running qr.qty & qr.qy benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  qr.qty(qr_out, b),
                  qr.qy(qr_out, b),
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


run_qr_benchmark(mat_size, replication, times)