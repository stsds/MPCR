\name{transpose}
\alias{transpose}
\alias{t,Rcpp_MMPR-method}
\title{transpose}
\usage{
\S4method{t}{Rcpp_MMPR}(x)
}
\arguments{
\item{x}{A MMPR vector/matrix.}
}
\value{
A float vector/matrix.
}
\description{
Transpose a MMPR vector/matrix.
}
\examples{
  \dontrun{
  library(MMPR)
  a <- matrix(1:20, nrow = 2)
  a_mmpr <- as.MMPR(a,2,10,"double")
  a_mmpr_transpose <- t(a_mmpr)
  }
}
