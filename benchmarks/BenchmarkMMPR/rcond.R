library(rbenchmark)
library(MMPR)


run_rcond_benchmark <- function (m,n,replication,times){

  # Create a random matrix of size n x n
  A <- matrix(rnorm(n^2), ncol = n)

  # Create a random matrix of size n x n
  B <- matrix(rnorm(n*m), ncol = n)

  cat("Matrix A : ")
  cat(paste(n,n,sep="*"))
  cat("\n")

  cat("Matrix B : ")
  cat(paste(m,n,sep="*"))
  cat("\n\n")


  mmpr_single_a <- as.MMPR(A,n,n,"single")
  mmpr_double_a <- as.MMPR(A,n,n,"double")


  mmpr_single_b <- as.MMPR(B,m,n,"single")
  mmpr_double_b <- as.MMPR(B,m,n,"double")

  cat("Running rcond bencmark")
  print(benchmark(replications=rep(replication,times),
                  rcond(mmpr_single_a),
                  rcond(mmpr_double_a),
                  rcond(mmpr_single_b),
                  rcond(mmpr_double_b),
                  columns=c("test", "replications", "elapsed")))

}

run_rcond_benchmark(100,30,10,3)