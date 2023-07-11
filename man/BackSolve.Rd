\name{forwardsolve}
\alias{forwardsolve}
\alias{backsolve}

\title{back-forwardsolve}

\description{
Solves a system of linear equations where the coefficient matrix is upper or lower triangular. The function solves the equation \code{aInputA aOutput = aInputB}, where \code{aInputA} is the coefficient matrix, \code{aOutput} is the solution vector, and \code{aInputB} is the right-hand side vector.
}

\usage{
 \S4method{backsolve}{Rcpp_MMPR,Rcpp_MMPR}(r, x, k = ncol(r), upper.tri = TRUE, transpose = FALSE)

 \S4method{forwardsolve}{Rcpp_MMPR,Rcpp_MMPR}(l, x, k = ncol(l), upper.tri = FALSE, transpose = FALSE)

}

\arguments{

\item{l}{A MMPR vector/matrix.}
\item{r}{A MMPR vector/matrix.}
\item{x}{A MMPR vector or a matrix whose columns give the right-hand sides for the equations.}
\item{k}{The number of columns of r and rows of x to use.}
\item{upper.tri}{logical; if TRUE , the upper triangular part of r is used. Otherwise, the lower one.}
\item{transpose}{logical; if true, solve  for t( l , r ) \%*\% output == x.}
}

\value{
  A MMPR matrix/vector representing the solution to the system of linear equations.
}

\examples{
\dontrun{
  library(MMPR)
  a <- matrix(c(2, 0, 0, 3), nrow = 2)
  b <- matrix(c(1, 2), nrow = 2)
  a_mmpr <- as.MMPR(a,2,2,"single")
  b_mmpr <- as.MMPR(b,2,1,"double")
  x <- forwardsolve(a_mmpr, b_mmpr)
  x
}
}

