\name{chol}
\alias{chol}
\alias{chol,Rcpp_MMPR-method}
\title{chol}
\usage{
\S4method{chol}{Rcpp_MMPR}(x,upper_triangle=TRUE)
}
\arguments{
\item{x}{A MMPR matrix.}
\item{upper_triangle}{boolean to check on which triangle the cholesky
 decomposition should be applied}
}
\value{
A MMPR matrix.
}
\description{
Cholesky factorization for a MMPR positive symmetric matrix.
}
\examples{
  \dontrun{
  library(MMPR)
  # x <- as.MMPR(vals,nrow,ncol,precision)
  chol_out <- chol(x)
  }
}
