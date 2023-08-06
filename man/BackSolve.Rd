\name{Back/Forward solve}
\alias{forwardsolve}
\alias{backsolve}

\title{Back/Forward solve}

\description{
Solves a system of linear equations where the coefficient matrix is upper or lower triangular. The function solves the equation \bold{\code{A X = B}}, where \code{A} is the coefficient matrix, \code{X} is the solution vector, and \code{B} is the right-hand side vector.
}

\usage{
 \S4method{backsolve}{Rcpp_MMPR,Rcpp_MMPR}(r, x, k = ncol(r), upper.tri = TRUE, transpose = FALSE)

 \S4method{forwardsolve}{Rcpp_MMPR,Rcpp_MMPR}(l, x, k = ncol(l), upper.tri = FALSE, transpose = FALSE)

}

\arguments{

\item{l}{An MMPR object.}
\item{r}{An MMPR object.}
\item{x}{An MMPR object whose columns give the right-hand sides for the equations.}
\item{k}{The number of columns of r and rows of x to use.}
\item{upper.tri}{logical; if TRUE, the upper triangular part of r is used. Otherwise, the lower one.}
\item{transpose}{logical; if TRUE, solve  for t( l , r ) \%*\% output == x.}
}

\value{
  An MMPR object represents the solution to the system of linear equations.
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

