library(rbenchmark)

run_eigen_becnhmark <- function(n,replication,times){
  cat("\n\n\n")

  cat("Matrix : ")
  cat(paste(n,n,sep="*"))
  cat("\n")

  matrix <- matrix(rnorm(n^2), ncol = n)

  cat("\n")
  print(benchmark(replications=rep(replication,times),
                 eigen(matrix),
                  columns=c("test", "replications", "elapsed")))

  cat("\n")
}

run_eigen_becnhmark(1000,100,3)