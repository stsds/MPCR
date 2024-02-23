\name{12-Extremes}
\alias{extremes}
\alias{min,Rcpp_MPCR-method}
\alias{max,Rcpp_MPCR-method}
\alias{which.min,Rcpp_MPCR-method}
\alias{which.max,Rcpp_MPCR-method}

\alias{MPCR.min}
\alias{MPCR.max}
\alias{MPCR.which.min}
\alias{MPCR.which.max}

\title{Min-Max Functions}
\description{
    Min-Max functions for MPCR objects values and indices, all NA values are disregarded.
}
\usage{
\S4method{min}{Rcpp_MPCR}(x)

\S4method{max}{Rcpp_MPCR}(x)

\S4method{which.min}{Rcpp_MPCR}(x)

\S4method{which.max}{Rcpp_MPCR}(x)
}
\arguments{
  \item{x}{
    An MPCR object.
  }
}
\value{
  Min/max value/index.
}
\examples{

  library(MPCR)

  x <- as.MPCR(1:20,precision="double")
  min <-min(x)
  min_idx <-which.min(x)

}