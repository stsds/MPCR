\name{isSymmetric}
\alias{isSymmetric}
\alias{isSymmetric,Rcpp_MMPR-method}
\title{isSymmetric}
\usage{
\S4method{isSymmetric}{Rcpp_MMPR}(object, ...)
}
\arguments{
\item{object}{A MMPR matrix.}

\item{...}{Ignored.}
}
\value{
A logical value.
}
\description{
Test if a MMPR matrix is symmetric.
}
\examples{
  \dontrun{
library(MMPR)

x <- as.MPPR(1:200,100,2,"Single")
isSymmetric(x)                      #false

crossprod_output<-crossprod(x)
isSymmetric(crossprod_output)       #true
}
}
