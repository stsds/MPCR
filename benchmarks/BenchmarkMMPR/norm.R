library(rbenchmark)
library(MMPR)


run_norm_benchmark <- function(n, replication, times) {

  A <- matrix(rnorm(n^2), ncol = n)

  mmpr_single <- as.MMPR(A, n, n, "single")
  mmpr_double <- as.MMPR(A, n, n, "double")

  cat("Matrix : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  cat("\n")
  print(benchmark(replications = rep(replication, times),
                  norm(mmpr_single, "O"),
                  norm(mmpr_single, "I"),
                  norm(mmpr_single, "F"),
                  norm(mmpr_single, "M"),
                  norm(mmpr_double, "O"),
                  norm(mmpr_double, "I"),
                  norm(mmpr_double, "F"),
                  norm(mmpr_double, "M"),
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