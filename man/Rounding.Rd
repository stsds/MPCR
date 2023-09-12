\name{18-Round}
\alias{round}
\alias{ceiling,Rcpp_MPCR-method}
\alias{floor,Rcpp_MPCR-method}
\alias{trunc,Rcpp_MPCR-method}
\alias{round,Rcpp_MPCR-method}
\title{Rounding functions}
\description{
    Rounding functions.
}
\usage{
\S4method{ceiling}{Rcpp_MPCR}(x)

\S4method{floor}{Rcpp_MPCR}(x)

\S4method{trunc}{Rcpp_MPCR}(x)

\S4method{round}{Rcpp_MPCR}(x, digits = 0)
}
\arguments{
  \item{x}{
    An MPCR object.
  }
  \item{digits}{
    The number of digits to use in rounding.
  }
}
\value{
  An MPCR object of the same dimensions as the input.
}
\examples{
  \dontrun{
  library(MPCR)

  x <- as.MPCR(1:20,precision="double")
  floor(x)
  }
}