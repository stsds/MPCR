\name{31-Symmetric}
\alias{isSymmetric}
\alias{isSymmetric,Rcpp_MPCR-method}
\alias{MPCR.isSymmetric}

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

library(MPCR)

x <- as.MPCR(1:50,25,2,"Single")
isSymmetric(x)                      #false

crossprod_output<-crossprod(x)
isSymmetric(crossprod_output)       #true

}
