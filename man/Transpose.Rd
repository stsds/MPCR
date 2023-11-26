\name{23-Transpose}
\alias{transpose}
\alias{MPCR.t}
\alias{t,Rcpp_MPCR-method}
\title{transpose}
\usage{
\S4method{t}{Rcpp_MPCR}(x)
}
\arguments{
\item{x}{An MPCR object.}
}
\value{
An MPCR object.
}
\description{
Transpose an MPCR object.
}
\examples{
  \donttest{
  library(MPCR)
  a <- matrix(1:20, nrow = 2)
  a_MPCR <- as.MPCR(a,2,10,"double")
  a_MPCR_transpose <- t(a_MPCR)
  }
}
