\name{Cholesky inverse}
\alias{chol2inv}
\alias{chol2inv,Rcpp_MMPR-method}
\title{cholesky inverse}
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
Performs the inverse of the original matrix using the Cholesky factorization of an MMPR matrix x.
}
\examples{
  \dontrun{
  library(MMPR)
  # x <- as.MMPR(vals,nrow,ncol,precision)
  chol_out <- chol(x)
  chol <- chol2inv(chol_out)
}
}
