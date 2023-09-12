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
  \dontrun{
  x <- chol(y,overwrite_input=FALSE,num_threads=8)
  x <- chol(x)
  }
}