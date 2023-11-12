\name{28-Cholesky inverse}
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
  \donttest{
  library(MPCR)
  x <- as.MPCR(c(1.21, 0.18, 0.13, 0.41, 0.06, 0.23,
                 0.18, 0.64, 0.10, -0.16, 0.23, 0.07,
                 0.13, 0.10, 0.36, -0.10, 0.03, 0.18,
                 0.41, -0.16, -0.10, 1.05, -0.29, -0.08,
                 0.06, 0.23, 0.03, -0.29, 1.71, -0.10,
                 0.23, 0.07, 0.18, -0.08, -0.10, 0.36),6,6,precision="single")
  chol_out <- chol(x)
  chol <- chol2inv(chol_out)
}
}
