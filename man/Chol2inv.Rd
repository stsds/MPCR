\name{29-Cholesky inverse}
\alias{chol2inv}
\alias{chol2inv,Rcpp_MPCR-method}
\title{cholesky inverse}
\usage{
\S4method{chol2inv}{Rcpp_MPCR}(x, size = NCOL(x))
}
\arguments{
\item{x}{An MPCR object.}

\item{size}{The number of columns to use.}

}
\value{
An MPCR object.
}
\description{
Performs the inverse of the original matrix using the Cholesky factorization of an MPCR matrix x.
}
\examples{
  \dontrun{
  library(MPCR)
  # x <- as.MPCR(vals,nrow,ncol,precision)
  chol_out <- chol(x)
  chol <- chol2inv(chol_out)
}
}
