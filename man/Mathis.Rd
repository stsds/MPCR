\name{mathis}
\alias{mathis}
\alias{is.finite,Rcpp_MMPR-method}
\alias{is.infinite,Rcpp_MMPR-method}
\alias{is.nan,Rcpp_MMPR-method}
\title{Finite, infinite, and NaNs}
\description{
    Finite, infinite, and NaNs.
}
\usage{
\S4method{is.finite}{Rcpp_MMPR}(x)

\S4method{is.infinite}{Rcpp_MMPR}(x)

\S4method{is.nan}{Rcpp_MMPR}(x)
}
\arguments{
  \item{x}{
    An MMPR object.
  }
}
\value{
  A bool vector/matrix of the same dimensions as the input.
}
\examples{
  \dontrun{
  library(MMPR)

  x <- as.MMPR(1:20,precision="double")
  is.nan(sqrt(x))
  }
}