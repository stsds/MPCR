\name{Special Math}
\alias{specialmath}
\alias{gamma,Rcpp_MMPR-method}
\alias{lgamma,Rcpp_MMPR-method}
\title{Special mathematical functions.}
\description{
Special mathematical functions.
}
\usage{
\S4method{gamma}{Rcpp_MMPR}(x)

\S4method{lgamma}{Rcpp_MMPR}(x)
}
\arguments{
\item{x}{
An MMPR object.
}
}
\value{
An MMPR object of the same dimensions as the input.
}
\examples{
\dontrun{
library(MMPR)

x <- as.MMPR(1:20,precision="double")
lgamma(x)
}
}