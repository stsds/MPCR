\name{29-Crossprod}
\alias{crossprod}
\alias{crossprod,Rcpp_MPCR-method}
\alias{tcrossprod,Rcpp_MPCR-method}
\title{crossprod}
\usage{
\S4method{crossprod}{Rcpp_MPCR}(x, y = NULL)

\S4method{tcrossprod}{Rcpp_MPCR}(x, y = NULL)

}
\arguments{
\item{x}{An MPCR object.}

\item{y}{Either \code{NULL}, or an MPCR matrix.}
}
\value{
An MPCR matrix.
}
\description{
Calculates the cross product of two MPCR matrices.
 It uses BLAS routine \code{gemm()} for \bold{A} X \bold{B} operations and \code{syrk()}
 for \bold{A} X \bold{A^T} operations.
}
\details{
 Calculates cross product of two MPCR matrices performs:
  \cr
  x \%*\% y  , t(x) \%*\% x
  \cr
  This function uses blas routine \code{gemm()} for \bold{A} X \bold{B} operations  & \code{syrk()} for \bold{A} X \bold{A^T} operations.
}
\examples{
    \donttest{
    library(MPCR)
    x <- as.MPCR(1:16,4,4,"single")
    y <- as.MPCR(1:20,4,5,"double")

    z <- crossprod(x)     # t(x) x
    z <- tcrossprod(x)    # x t(x)
    z <- crossprod(x,y)   # x y
    z <- x \%*\% y          # x y
    }
}