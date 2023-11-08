\name{13-Mathis}
\alias{mathis}
\alias{is.finite,Rcpp_MPCR-method}
\alias{is.infinite,Rcpp_MPCR-method}
\alias{is.nan,Rcpp_MPCR-method}
\title{Finite, infinite, and NaNs}
\description{
    Finite, infinite, and NaNs.
}
\usage{
\S4method{is.finite}{Rcpp_MPCR}(x)

\S4method{is.infinite}{Rcpp_MPCR}(x)

\S4method{is.nan}{Rcpp_MPCR}(x)
}
\arguments{
  \item{x}{
    An MPCR object.
  }
}
\value{
  A bool vector/matrix of the same dimensions as the input.
}
\examples{
  \donttest{
  library(MPCR)

  x <- as.MPCR(1:20,precision="double")
  is.nan(sqrt(x))
  }
}