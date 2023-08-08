library(rbenchmark)
library(MMPR)


run_transpose_benchmark <- function(m, n, replication, times) {

  A <- matrix(rnorm(n * m), ncol = n)

  mmpr_single <- as.MMPR(A, m, n, "single")
  mmpr_double <- as.MMPR(A, m, n, "double")

  cat("\n\n\n")
  cat("Running transpose \n")

  print(benchmark(replications = rep(replication, times),
                  t(mmpr_single),
                  t(mmpr_double),
                  t(A),
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

run_transpose_benchmark(row, col, replication, times)