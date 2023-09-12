\name{28-Cholesky decomposition}
\alias{chol}
\alias{chol,Rcpp_MPCR-method}
\title{cholesky decomposition}
\usage{
\S4method{chol}{Rcpp_MPCR}(x,upper_triangle=TRUE)
}
\arguments{
\item{x}{An MPCR matrix.}
\item{upper_triangle}{Boolean to check on which triangle the cholesky
 decomposition should be applied.}
}
\value{
An MPCR matrix.
}
\description{
Performs the Cholesky factorization of a positive definite MPCR matrix x.
}
\examples{
  \dontrun{
  library(MPCR)
  # x <- as.MPCR(vals,nrow,ncol,precision)
  chol_out <- chol(x)
  }
}
