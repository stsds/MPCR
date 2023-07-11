\name{chol2inv}
\alias{chol2inv}
\alias{chol2inv,Rcpp_MMPR-method}
\title{chol2inv}
\usage{
\S4method{chol2inv}{Rcpp_MMPR}(x, size = NCOL(x))
}
\arguments{
\item{x}{An MMPR object.}

\item{size}{The number of columns to use.}

}
\value{
An MMPR object.
}
\description{
Return the inverse of the original matrix using the cholesky factorization of
an MMPR matrix.
}
\examples{
  \dontrun{
  library(MMPR)
  # x <- as.MMPR(vals,nrow,ncol,precision)
  chol_out <- chol(x)
  chol <- chol2inv(chol_out)
}
}
