library(rbenchmark)


run_crossprod_benchmark <- function(row, col, replication, times) {
  cat("\n\n\n")

  cat("Matrix 1 : ")
  cat(paste(row, col, sep = "*"))
  cat("\n")

  cat("Matrix 2 : ")
  cat(paste(col, row, sep = "*"))
  cat("\n")
  matrix_1 <- matrix(runif(row * col, min = 0.5, max = 2), nrow = row, ncol = col)
  matrix_2 <- matrix(runif(row * col, min = -3, max = 2), nrow = col, ncol = row)
  matrix_3 <- matrix(runif(row * col, min = -3, max = 2), nrow = row, ncol = col)


  cat("\n")
  cat("\n\n")
  cat("Running crossprod benchmark with 2 input \n")
  print(benchmark(replications = rep(replication, times),
                  (matrix_1 %*% matrix_2),
                  columns = c("test", "replications", "elapsed")))

  cat("Running tcrossprod benchmark with 2 input \n")
  print(benchmark(replications = rep(replication, times),
                  tcrossprod(matrix_1, matrix_3),
                  columns = c("test", "replications", "elapsed")))


  cat("\n\n\n")
  matrix_3 <- matrix(runif(row * row, min = -3, max = 2), nrow = row, ncol = row)


  cat("\n")
  cat("Running crossprod benchmark with 1 input \n")
  print(benchmark(replications = rep(replication, times),
                  crossprod(matrix_3, y = NULL),
                  columns = c("test", "replications", "elapsed")))

  cat("\n")
  cat("Running tcrossprod benchmark with 1 input \n")
  print(benchmark(replications = rep(replication, times),
                  tcrossprod(matrix_3, y = NULL),
                  columns = c("test", "replications", "elapsed")))

  cat("\n")
}

# Define the arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 4) {
  cat("\n\n\n\n")
  stop("Please provide correct arguments, 1-row 2-col 3-number_of_replication 4-times")
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

run_crossprod_benchmark(row, col, replication, times)






