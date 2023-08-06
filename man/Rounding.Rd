\name{18-Round}
\alias{round}
\alias{ceiling,Rcpp_MMPR-method}
\alias{floor,Rcpp_MMPR-method}
\alias{trunc,Rcpp_MMPR-method}
\alias{round,Rcpp_MMPR-method}
\title{Rounding functions}
\description{
    Rounding functions.
}
\usage{
\S4method{ceiling}{Rcpp_MMPR}(x)

\S4method{floor}{Rcpp_MMPR}(x)

\S4method{trunc}{Rcpp_MMPR}(x)

\S4method{round}{Rcpp_MMPR}(x, digits = 0)
}
\arguments{
  \item{x}{
    An MMPR object.
  }
  \item{digits}{
    The number of digits to use in rounding.
  }
}
\value{
  An MMPR object of the same dimensions as the input.
}
\examples{
  \dontrun{
  library(MMPR)

  x <- as.MMPR(1:20,precision="double")
  floor(x)
  }
}