\name{28-Cholesky decomposition}
\alias{chol}
\alias{chol,Rcpp_MMPR-method}
\title{cholesky decomposition}
\usage{
\S4method{chol}{Rcpp_MMPR}(x,upper_triangle=TRUE)
}
\arguments{
\item{x}{An MMPR matrix.}
\item{upper_triangle}{Boolean to check on which triangle the cholesky
 decomposition should be applied.}
}
\value{
An MMPR matrix.
}
\description{
Performs the Cholesky factorization of a positive definite MMPR matrix x.
}
\examples{
  \dontrun{
  library(MMPR)
  # x <- as.MMPR(vals,nrow,ncol,precision)
  chol_out <- chol(x)
  }
}
