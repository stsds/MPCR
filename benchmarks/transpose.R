library(rbenchmark)
library(MMPR)



run_issymmeric_benchmark <- function (m,n,replication,times){

  A <- matrix(rnorm(n*m), ncol = n)

  mmpr_single <- as.MMPR(A,m,n,"single")
  mmpr_double <- as.MMPR(A,m,n,"double")

  cat("\n\n\n")
  cat("Running transpose \n")

  print(benchmark(replications=rep(replication,times),
                  t(mmpr_single),
                  t(mmpr_double),
                  t(A),
                  columns=c("test", "replications", "elapsed")))

}

run_issymmeric_benchmark(10000,100,3)