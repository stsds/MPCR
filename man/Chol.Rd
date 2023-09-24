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
  \donttest{
  library(MPCR)
  x <- as.MPCR(c(1.21, 0.18, 0.13, 0.41, 0.06, 0.23,
                 0.18, 0.64, 0.10, -0.16, 0.23, 0.07,
                 0.13, 0.10, 0.36, -0.10, 0.03, 0.18,
                 0.41, -0.16, -0.10, 1.05, -0.29, -0.08,
                 0.06, 0.23, 0.03, -0.29, 1.71, -0.10,
                 0.23, 0.07, 0.18, -0.08, -0.10, 0.36),6,6,precision="double")
  chol_out <- chol(x)
  }
}
