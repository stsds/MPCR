\name{MMPRTile Chol}
\alias{MMPRTile Chol}
\alias{MMPRTile Chol,Rcpp_MMPRTile-method}
\title{MMPRTile Chol  ( Cholesky decomposition )}
\description{
    Calculate cholesky decomposition for positive tiled-symmetric matrix.
}
\usage{

\S4method{chol}{Rcpp_MMPR}(x, overwrite_input = TRUE, num_threads = 1)

}
\arguments{
  \item{x}{
    MMPR tile matrix.
  }
  \item{overwrite_input}{
    Flag to determine whether to overwrite the input ( TRUE ), or return a new MMPR tile matrix.
  }
  \item{num_threads}{
    Integer to determine number if thread to run using openmp, \bold{default} = 1 (serial with no parallelization).
  }
}
\value{
  MMPR tile matrix.
}
\examples{
  \dontrun{
  x <- chol(y,overwrite_input=FALSE,num_threads=8)
  x <- chol(x)
  }
}