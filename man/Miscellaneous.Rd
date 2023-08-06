\name{Miscmath}
\alias{miscmath}
\alias{abs,Rcpp_MMPR-method}
\alias{sqrt,Rcpp_MMPR-method}
\title{Miscellaneous mathematical functions}
\description{
Miscellaneous mathematical functions.
}
\usage{
\S4method{abs}{Rcpp_MMPR}(x)

\S4method{sqrt}{Rcpp_MMPR}(x)
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
sqrt(x)
}
}