\name{MMPRTile Chol}
\alias{MMPRTile Chol}
\alias{MMPRTile Chol,Rcpp_MMPRTile-method}
\title{MMPRTile Chol  ( Cholesky decomposition )}
\description{
    Calculate Cholesky decomposition for Positive Tiled-Symmetric Matrix
}
\usage{

\S4method{chol}{Rcpp_MMPR}(x, overwrite_input = TRUE, num_threads = 1)

}
\arguments{
  \item{x}{
    MMPR Tile Matrix
  }
  \item{overwrite_input}{
    Flag to determine whether to overwrite the input ( TRUE ) , or return a new MMPR Tile Matrix
  }
  \item{num_threads}{
    Integer to determine number if thread to run using openmp ,default = 1 (serial with no parallelization)
  }
}
\value{
  MMPR Tile Matrix
}
\examples{
  \dontrun{
  x <- chol(y,overwrite_input=FALSE,num_threads=8)
  x <- chol(x)
  }
}