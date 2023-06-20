library(rbenchmark)
library(MMPR)


run_qr_benchmark <- function(n,replication,times){

  # Generate a random matrix of size n x n
  set.seed(123)
  matrix <- matrix(rnorm(n*n), nrow = n)

  cat("Matrix A : ")
  cat(paste(n,n,sep="*"))

  mmpr_matrix_single <- as.MMPR(matrix, n, n, "single")
  mmpr_matrix_double <- as.MMPR(matrix, n, n, "double")

  cat("\n\n\n")
  cat("Running qr benchmark \n")
  print(benchmark(replications=rep(replication,times),
                  qr(mmpr_matrix_single),
                  qr(mmpr_matrix_double),
                  columns=c("test", "replications", "elapsed")))


  qr_single <- qr(mmpr_matrix_single)
  qr_double <- qr(mmpr_matrix_double)

  print(class(qr_single))

  cat("Running qr.R benchmark \n")
  print(benchmark(replications=rep(replication,times),
                  qr.R(qr_single),
                  qr.R(qr_double),
                  columns=c("test", "replications", "elapsed")))


  cat("Running qr.Q benchmark \n")
  print(benchmark(replications=rep(replication,times),
                  qr.Q(qr_single),
                  qr.Q(qr_double),
                  columns=c("test", "replications", "elapsed")))


  b <- rnorm(n)

  mmpr_random_single <- as.MMPR(b,precision="single")
  mmpr_random_double <- as.MMPR(b,precision="double")

  cat("Running qr.qy & qr.qty benchmark \n")
  print(benchmark(replications=rep(replication,times),
                  qr.qty(qr_single,mmpr_random_single),
                  qr.qy(qr_double,mmpr_random_double),
                  columns=c("test", "replications", "elapsed")))



}

run_qr_benchmark(100,100,3)