\name{chol2inv}
\alias{chol2inv}
\alias{chol2inv,Rcpp_MMPR-method}
\title{chol2inv}
\usage{
\S4method{chol2inv}{Rcpp_MMPR}(x, size = NCOL(x))
}
\arguments{
\item{x}{A MMPR vector/matrix.}

\item{size}{The number of columns to use.}

}
\value{
A MMPR vector/matrix.
}
\description{
Return the inverse of the original matrix using the Cholesky factorization of
a MMPR matrix.
}
\examples{
  \dontrun{
  library(MMPR)
  # x <- as.MMPR(vals,nrow,ncol,precision)
  chol_out <- chol(x)
  chol <- chol2inv(chol_out)
}
}
