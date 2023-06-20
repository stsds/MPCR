library(rbenchmark)

run_svd_benchmark <- function (n,replication,times){

  cat("Matrix A : ")
  cat(paste(n,n,sep="*"))
  cat("\n")

  # Create a random matrix of size n x n
  A <- matrix(rnorm(n^2), ncol = n)



  cat("\n\n")
  cat("Running svd benchmark \n")
  print(benchmark(replications=rep(replication,times),
                  svd(A),
                  columns=c("test", "replications", "elapsed")))

  cat("\n\n")
  cat("Running La.svd benchmark \n")
  print(benchmark(replications=rep(replication,times),
                  La.svd(A),
                  columns=c("test", "replications", "elapsed")))


}

run_svd_benchmark(100,100,3)