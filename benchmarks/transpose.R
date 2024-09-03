library(rbenchmark)
library(MPCR)


run_transpose_benchmark <- function(m, n, replication, times,operation_placement) {

  A <- matrix(rnorm(n * m), ncol = n)

  MPCR_single <- as.MPCR(A, m, n, "single",operation_placement)
  MPCR_double <- as.MPCR(A, m, n, "double",operation_placement)

  cat("\n\n\n")
  cat("Running transpose \n")
  MPCR.SetOperationPlacement(operation_placement)

  print(benchmark(replications = rep(replication, times),
                  t(MPCR_single),
                  t(MPCR_double),
                  t(A),
                  columns = c("test", "replications", "elapsed")))

}

# Define the arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 5) {
  cat("\n\n\n\n")
  stop("Please provide correct arguments, 1-row 2-col 3-number_of_replication 4-times 5-operation_placement")
}

row <- as.integer(args[1])
col <- as.integer(args[2])
replication <- as.integer(args[3])
times <- as.integer(args[4])
operation_placement <- toString(args[5])


cat("Matrix size : ")
cat(paste(row, col, sep = "*"))
cat("\n")
cat("replication : ")
cat(replication)
cat("times : ")
cat(times)
cat("\n")
cat("Operation Placement : ")
cat(operation_placement)
cat("\n")

run_transpose_benchmark(row, col, replication, times,operation_placement)