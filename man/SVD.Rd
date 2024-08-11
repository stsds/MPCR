\name{36-Singular value decomposition}
\alias{svd}
\alias{La.svd,Rcpp_MPCR-method}
\alias{svd,Rcpp_MPCR-method}

\alias{MPCR.La.svd}
\alias{MPCR.svd}

\title{SVD}
\usage{
\S4method{La.svd}{Rcpp_MPCR}(x, nu = min(n, p), nv = min(n, p))

\S4method{svd}{Rcpp_MPCR}(x, nu = min(n, p), nv = min(n, p))
}
\arguments{
\item{x}{An MPCR matrix.}

\item{nu, nv}{The number of left/right singular vectors to return.}

}
\description{
SVD factorization.
}
\details{
The factorization is performed by the LAPACK routine \code{gesdd()}.
}

\value{
The SVD decomposition of the MPCR matrix.
}

\examples{
\donttest{
library(MPCR)
svd_vals <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1)

x <- as.MPCR(svd_vals,9,4,"single")
y <- svd(x)
}
}