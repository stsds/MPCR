\name{32-Symmetric}
\alias{isSymmetric}
\alias{isSymmetric,Rcpp_MPCR-method}
\title{isSymmetric}
\usage{
\S4method{isSymmetric}{Rcpp_MPCR}(object, ...)
}
\arguments{
\item{object}{An MPCR matrix.}

\item{...}{Ignored.}
}
\value{
A logical value.
}
\description{
Check if a given MPCR matrix is symmetric.
}
\examples{
  \dontrun{
library(MPCR)

x <- as.MPPR(1:200,100,2,"Single")
isSymmetric(x)                      #false

crossprod_output<-crossprod(x)
isSymmetric(crossprod_output)       #true
}
}
