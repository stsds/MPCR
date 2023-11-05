\name{42-MPCRTile POTRF}
\alias{MPCRTile Chol}
\alias{chol,Rcpp_MPCRTile-method}
\title{MPCRTile Chol  ( Cholesky decomposition )}
\description{
    Tile-based Cholesky decomposition of a positive definite tile-based symmetric matrix.
}
\usage{

\S4method{chol}{Rcpp_MPCRTile}(x, overwrite_input = TRUE, num_threads = 1)

}
\arguments{
  \item{x}{
    An MPCR tile matrix.
  }
  \item{overwrite_input}{
    A flag to determine whether to overwrite the input ( TRUE ), or return a new MPCR tile matrix.
  }
  \item{num_threads}{
    An integer to determine number if thread to run using openmp, default = 1 (serial with no parallelization).
  }
}
\value{
  An MPCR tile matrix.
}
\examples{
\donttest{
 library(MPCR)
  a <- matrix(c(1.21, 0.18, 0.13, 0.41, 0.06, 0.23,
              0.18, 0.64, 0.10, -0.16, 0.23, 0.07,
              0.13, 0.10, 0.36, -0.10, 0.03, 0.18,
              0.41, -0.16, -0.10, 1.05, -0.29, -0.08,
              0.06, 0.23, 0.03, -0.29, 1.71, -0.10,
              0.23, 0.07, 0.18, -0.08, -0.10, 0.36), 6, 6)
  b <- c("float", "double", "float", "float",
       "double", "double", "float", "float",
       "double")

  chol_mat <- new(MPCRTile, 6, 6, 2, 2, a, b)

  x <- chol(chol_mat,overwrite_input=FALSE,num_threads=8)
  print(chol_mat)
  print(x)

  chol(chol_mat)
  print(chol_mat)

  }
}