library(rbenchmark)


run_norm_benchmark <- function(n, replication, times) {

  A <- matrix(rnorm(n^2), ncol = n)

  cat("Matrix : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  cat("\n")
  print(benchmark(replications = rep(replication, times),
                  norm(A, "O"),
                  norm(A, "I"),
                  norm(A, "F"),
                  norm(A, "M"),
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

run_norm_benchmark(mat_size, replication, times)