\name{extremes}
\alias{extremes}
\alias{min,Rcpp_MMPR-method}
\alias{max,Rcpp_MMPR-method}
\alias{which.min,Rcpp_MMPR-method}
\alias{which.max,Rcpp_MMPR-method}
\title{Min-Max Functions}
\description{
    Min-Max Functions for matrix/vector values and indices , all NA values are disregarded
}
\usage{
\S4method{min}{Rcpp_MMPR}(x)

\S4method{max}{Rcpp_MMPR}(x)

\S4method{which.min}{Rcpp_MMPR}(x)

\S4method{which.max}{Rcpp_MMPR}(x)
}
\arguments{
  \item{x}{
    A MMPR vector/matrix.
  }
}
\value{
  Min/Max value/Index
}
\examples{
  \dontrun{
  library(MMPR)

  x <- as.MMPR(1:20,precision="double")
  min <-min(x)
  min_idx <-which.min(x)
  }
}