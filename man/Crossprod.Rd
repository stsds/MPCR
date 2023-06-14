\name{crossprod}
\alias{crossprod}
\alias{crossprod,Rcpp_MMPR-method}
\alias{tcrossprod,Rcpp_MMPR-method}
\title{crossprod}
\usage{
\S4method{crossprod}{Rcpp_MMPR}(x, y = NULL)

\S4method{tcrossprod}{Rcpp_MMPR}(x, y = NULL)

}
\arguments{
\item{x}{A MMPR vector/matrix.}

\item{y}{Either \code{NULL}, or a MMPR matrix.}
}
\value{
A MMPR matrix.
}
\description{
This function calculates the cross product of two MMPR matrices.
 It uses BLAS routine \code{gemm()} for A B operations and \code{syrk()}
 for A A^T operations.
}
\details{
 Calculate CrossProduct of 2 MPR Matrices performs:
  x %*% y  , t(x) %*% x
  this function uses blas routine \code{gemm()} for A B operations  & \code{syrk()} for A A^T operations
}
\examples{
    \dontrun{
    library(MMPR)
    x <- as.MMPR(1:16,4,4,"single")
    y <- as.MMPR(1:20,4,5,"double")

    z <- crossprod(x)     # t(x) x
    z <- tcrossprod(x)    # x t(x)
    z <- crossprod(x,y)   # x y
    z <- x %*% y          # x y
    }
}