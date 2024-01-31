\name{36-Solve}
\alias{solve}
\alias{MPCR.solve}
\alias{solve,Rcpp_MPCR-method}
\title{solve}
\usage{
\S4method{solve}{Rcpp_MPCR}(a, b = NULL, ...)
}
\arguments{
\item{a, b}{An MPCR objects.}

\item{...}{Ignored.}
}
\value{
Solves the equation AX=B .and if B=NULL t(A) will be used.
}
\description{
Solve a system of equations or invert an MPCR matrix.
}
\examples{
\donttest{
library(MPCR)

x <- as.MPCR(1:20,4,5,"double")
solve(x)
}
}
