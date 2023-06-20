library(rbenchmark)
library(MMPR)

run_eigen_becnhmark <- function(n,replication,times){
  cat("\n\n\n")

  cat("Matrix : ")
  cat(paste(n,n,sep="*"))
  cat("\n")

  matrix <- matrix(rnorm(n^2), ncol = n)

  mmpr_single <- as.MMPR(matrix,n,n,"single")
  mmpr_double <- as.MMPR(matrix,n,n,"double")

  cat("\n")
  print(benchmark(replications=rep(replication,times),
                  eigen(mmpr_single),
                  eigen(mmpr_double),
                  columns=c("test", "replications", "elapsed")))

  cat("\n")
}

run_eigen_becnhmark(1000,100,3)