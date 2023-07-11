\name{chol}
\alias{chol}
\alias{chol,Rcpp_MMPR-method}
\title{chol}
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
Cholesky factorization for an MMPR positive symmetric matrix.
}
\examples{
  \dontrun{
  library(MMPR)
  # x <- as.MMPR(vals,nrow,ncol,precision)
  chol_out <- chol(x)
  }
}
