library(rbenchmark)


run_norm_benchmark <- function(n,replication,times){

  A <- matrix(rnorm(n^2), ncol = n)

  cat("Matrix : ")
  cat(paste(n,n,sep="*"))
  cat("\n")

  cat("\n")
  print(benchmark(replications=rep(replication,times),
                  norm(A,"O"),
                  norm(A,"I"),
                  norm(A,"F"),
                  norm(A,"M"),
                  columns=c("test", "replications", "elapsed")))

}

run_norm_benchmark(100,100,3)