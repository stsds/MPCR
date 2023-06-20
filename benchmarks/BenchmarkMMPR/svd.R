library(rbenchmark)
library(MMPR)


run_svd_benchmark <- function(n, replication, times) {

  cat("Matrix A : ")
  cat(paste(n, n, sep = "*"))
  cat("\n")

  # Create a random matrix of size n x n
  A <- matrix(rnorm(n^2), ncol = n)
  mmpr_single <- as.MMPR(A,n,n,"single")
  mmpr_double <- as.MMPR(A,n,n,"double")


  cat("\n\n")
  cat("Running svd benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  svd(mmpr_single),
                  svd(mmpr_double),
                  columns = c("test", "replications", "elapsed")))

  cat("\n\n")
  cat("Running La.svd benchmark \n")
  print(benchmark(replications = rep(replication, times),
                  La.svd(mmpr_single),
                  La.svd(mmpr_double),
                  columns = c("test", "replications", "elapsed")))


}

run_svd_benchmark(100, 100, 3)


