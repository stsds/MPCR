\name{37-Singular value decomposition}
\alias{svd}
\alias{La.svd,Rcpp_MMPR-method}
\alias{svd,Rcpp_MMPR-method}
\title{SVD}
\usage{
\S4method{La.svd}{Rcpp_MMPR}(x, nu = min(n, p), nv = min(n, p))

\S4method{svd}{Rcpp_MMPR}(x, nu = min(n, p), nv = min(n, p))
}
\arguments{
\item{x}{An MMPR matrix.}

\item{nu, nv}{The number of left/right singular vectors to return.}

}
\description{
SVD factorization.
}
\details{
The factorization is performed by the LAPACK routine \code{gesdd()}.
}
\examples{
\dontrun{
library(MMPR)
x <- as.MMPR("your_data",nrow,ncol,precision)
y <- svd(x)
}
}