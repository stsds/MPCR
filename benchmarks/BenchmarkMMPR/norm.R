library(rbenchmark)
library(MMPR)


run_norm_benchmark <- function(n,replication,times){

  A <- matrix(rnorm(n^2), ncol = n)

  mmpr_single <- as.MMPR(A,n,n,"single")
  mmpr_double <- as.MMPR(A,n,n,"double")

  cat("Matrix : ")
  cat(paste(n,n,sep="*"))
  cat("\n")

  cat("\n")
  print(benchmark(replications=rep(replication,times),
                  norm(mmpr_single,"O"),
                  norm(mmpr_single,"I"),
                  norm(mmpr_single,"F"),
                  norm(mmpr_single,"M"),
                  norm(mmpr_double,"O"),
                  norm(mmpr_double,"I"),
                  norm(mmpr_double,"F"),
                  norm(mmpr_double,"M"),
                  columns=c("test", "replications", "elapsed")))

}

run_norm_benchmark(100,100,3)