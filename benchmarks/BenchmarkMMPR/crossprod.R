library(rbenchmark)
library(MMPR)


run_gemm_benchmark <- function(row, col, replication, times) {
  cat("\n\n\n")

  cat("Matrix 1 : ")
  cat(paste(row, col, sep = "*"))
  cat("\n")

  cat("Matrix 2 : ")
  cat(paste(col, row, sep = "*"))
  cat("\n")
  matrix_1 <- matrix(runif(row * col, min = 0.5, max = 2), nrow = row, ncol = col)
  matrix_2 <- matrix(runif(row * col, min = -3, max = 2), nrow = col, ncol = row)
  print(nrow(matrix_1))
  print(ncol(matrix_1))
  print(nrow(matrix_2))
  print(ncol(matrix_2))

  mmpr_matrix_single_1 <- as.MMPR(matrix_1, row, col, "single")
  mmpr_matrix_double_1 <- as.MMPR(matrix_1, row, col, "double")


  mmpr_matrix_single_2 <- as.MMPR(matrix_2, col, row, "single")
  mmpr_matrix_double_2 <- as.MMPR(matrix_2, col, row, "double")


  cat("\n\n")
  cat("Running crossprod benchmark with 2 input \n")
  print(benchmark(replications = rep(replication, times),
                  crossprod(mmpr_matrix_single_1, mmpr_matrix_single_2),
                  crossprod(mmpr_matrix_double_1, mmpr_matrix_double_2),
                  columns = c("test", "replications", "elapsed")))

  mmpr_matrix_single_3 <- as.MMPR(matrix_2, row, col, "single")
  mmpr_matrix_double_3 <- as.MMPR(matrix_2, row, col, "double")

  cat("\n\n")
  cat("Running tcrossprod benchmark with 2 input \n")
  print(benchmark(replications = rep(replication, times),
                  tcrossprod(mmpr_matrix_single_1, mmpr_matrix_single_3),
                  tcrossprod(mmpr_matrix_double_1, mmpr_matrix_double_3),
                  columns = c("test", "replications", "elapsed")))

  cat("\n\n\n")
  matrix_3 <- matrix(runif(row * row, min = -3, max = 2), nrow = row, ncol = row)

  mmpr_matrix_single_3 <- as.MMPR(matrix_3, row, row, "single")
  mmpr_matrix_double_3 <- as.MMPR(matrix_3, row, row, "double")
  cat("Matrix 3 : ")
  cat(paste(row, row, sep = "*"))
  cat("\n")
  cat("\n\n\n")
  cat("Running crossprod benchmark with 1 input \n")
  print(benchmark(replications = rep(replication, times),
                  crossprod(mmpr_matrix_single_3),
                  crossprod(mmpr_matrix_double_3),
                  columns = c("test", "replications", "elapsed")))


  cat("Running tcrossprod benchmark with 1 input \n")
  print(benchmark(replications = rep(replication, times),
                  tcrossprod(mmpr_matrix_single_3),
                  tcrossprod(mmpr_matrix_double_3),
                  columns = c("test", "replications", "elapsed")))


  cat("\n")
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

run_gemm_benchmark(row, col, replication, times)






