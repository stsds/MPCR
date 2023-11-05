\name{38-Back/Forward solve}
\alias{forwardsolve,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{backsolve,Rcpp_MPCR,Rcpp_MPCR-method}

\title{Back/Forward solve}

\description{
Solves a system of linear equations where the coefficient matrix is upper or lower triangular. The function solves the equation \bold{\code{A X = B}}, where \code{A} is the coefficient matrix, \code{X} is the solution vector, and \code{B} is the right-hand side vector.
}

\usage{
 \S4method{backsolve}{Rcpp_MPCR,Rcpp_MPCR}(r, x, k = ncol(r), upper.tri = TRUE, transpose = FALSE)

 \S4method{forwardsolve}{Rcpp_MPCR,Rcpp_MPCR}(l, x, k = ncol(l), upper.tri = FALSE, transpose = FALSE)

}

\arguments{

\item{l}{An MPCR object.}
\item{r}{An MPCR object.}
\item{x}{An MPCR object whose columns give the right-hand sides for the equations.}
\item{k}{The number of columns of r and rows of x to use.}
\item{upper.tri}{logical; if TRUE, the upper triangular part of r is used. Otherwise, the lower one.}
\item{transpose}{logical; if TRUE, solve  for t( l , r ) \%*\% output == x.}
}

\value{
  An MPCR object represents the solution to the system of linear equations.
}

\examples{
\donttest{
  library(MPCR)
  a <- matrix(c(2, 0, 0, 3), nrow = 2)
  b <- matrix(c(1, 2), nrow = 2)
  a_MPCR <- as.MPCR(a,2,2,"single")
  b_MPCR <- as.MPCR(b,2,1,"double")
  x <- forwardsolve(a_MPCR, b_MPCR)
  x
}
}

