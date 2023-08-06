\name{Crossprod}
\alias{crossprod}
\alias{crossprod,Rcpp_MMPR-method}
\alias{tcrossprod,Rcpp_MMPR-method}
\title{crossprod}
\usage{
\S4method{crossprod}{Rcpp_MMPR}(x, y = NULL)

\S4method{tcrossprod}{Rcpp_MMPR}(x, y = NULL)

}
\arguments{
\item{x}{An MMPR object.}

\item{y}{Either \code{NULL}, or an MMPR matrix.}
}
\value{
An MMPR matrix.
}
\description{
Calculates the cross product of two MMPR matrices.
 It uses BLAS routine \code{gemm()} for \bold{A} X \bold{B} operations and \code{syrk()}
 for \bold{A} X \bold{A^T} operations.
}
\details{
 Calculates cross product of two MMPR matrices performs:
  \cr
  x \%*\% y  , t(x) \%*\% x
  \cr
  This function uses blas routine \code{gemm()} for \bold{A} X \bold{B} operations  & \code{syrk()} for \bold{A} X \bold{A^T} operations.
}
\examples{
    \dontrun{
    library(MMPR)
    x <- as.MMPR(1:16,4,4,"single")
    y <- as.MMPR(1:20,4,5,"double")

    z <- crossprod(x)     # t(x) x
    z <- tcrossprod(x)    # x t(x)
    z <- crossprod(x,y)   # x y
    z <- x \%*\% y          # x y
    }
}