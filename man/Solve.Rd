\name{solve}
\alias{solve}
\alias{solve,Rcpp_MMPR-method}
\title{solve}
\usage{
\S4method{solve}{Rcpp_MMPR}(a, b = NULL, ...)
}
\arguments{
\item{a, b}{A MMPR vector/matrix.}

\item{...}{Ignored.}
}
\value{
Solves the equation AX=B .and if B=NULL t(A) will be used.
}
\description{
Solve a system of equations or invert a MMPR matrix.
}
\examples{
\dontrun{
library(MMPR)

x <- as.MMPR(1:20,4,5,"double")
y <- crossprod(x)
solve(y)

}
}
