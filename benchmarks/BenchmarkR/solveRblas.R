library(rbenchmark)

run_solve_benchmark <- function (n,replication,times){

  # Create a random matrix
  set.seed(123)
  A <- matrix(rnorm(n*n), nrow = n, ncol = n)

  # Create a random vector
  b <- rnorm(n)


  cat("Matrix A : ")
  cat(paste(n,n,sep="*"))
  cat("\n")

  cat("Matrix B : ")
  cat(n)
  cat("\n")

  cat("\n\n")
  cat("Running solve benchmark \n")
  print(benchmark(replications=rep(replication,times),
                  solve(A,b),
                  columns=c("test", "replications", "elapsed")))

}

run_solve_benchmark(1000,100,3)

