library(rbenchmark)


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

  cat("Running rcond bencmark")
  print(benchmark(replications=rep(replication,times),
                  rcond(A),
                  rcond(B),
                  columns=c("test", "replications", "elapsed")))

}

run_rcond_benchmark(100,30,10,3)