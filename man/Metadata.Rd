\name{Metadata}
\alias{metadata}
\alias{storage.mode,Rcpp_MMPR-method}
\alias{typeof,Rcpp_MMPR-method}
\title{Metadata functions}
\description{
    Metadata functions.
}
\usage{
\S4method{storage.mode}{Rcpp_MMPR}(x)
\S4method{typeof}{Rcpp_MMPR}(x)
\S4method{MMPR.object.size}{Rcpp_MMPR}(x)
\S4method{MMPR.ChangePrecision}{Rcpp_MMPR}(x,precision)

}
\arguments{
  \item{x}{
    An MMPR object.
  }
  \item{precision}{
      String with the required precision.
    }
}
\value{
  Prints/change metadata about an MMPR object.
}
\examples{
  \dontrun{
  library(MMPR)

  x <- as.MMPR(1:20,precision="double")
  typeof(x)
  MMPR.ChangePrecision(x,"single")
  }
}