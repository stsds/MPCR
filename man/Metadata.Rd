\name{26-Metadata}
\alias{metadata}
\alias{storage.mode,Rcpp_MPCR-method}
\alias{typeof,Rcpp_MPCR-method}
\alias{MPCR.ChangePrecision,Rcpp_MPCR-method}
\alias{MPCR.object.size,Rcpp_MPCR-method}
\title{Metadata functions}
\description{
    Metadata functions.
}
\usage{
\S4method{storage.mode}{Rcpp_MPCR}(x)
\S4method{typeof}{Rcpp_MPCR}(x)
\S4method{MPCR.object.size}{Rcpp_MPCR}(x)
\S4method{MPCR.ChangePrecision}{Rcpp_MPCR}(x,precision)

}
\arguments{
  \item{x}{
    An MPCR object.
  }
  \item{precision}{
      String with the required precision.
    }
}
\value{
  Prints/change metadata about an MPCR object.
}
\examples{
  \dontrun{
  library(MPCR)

  x <- as.MPCR(1:20,precision="double")
  typeof(x)
  MPCR.ChangePrecision(x,"single")
  }
}