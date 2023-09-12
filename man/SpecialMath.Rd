\name{21-Special Math}
\alias{specialmath}
\alias{gamma,Rcpp_MPCR-method}
\alias{lgamma,Rcpp_MPCR-method}
\title{Special mathematical functions.}
\description{
Special mathematical functions.
}
\usage{
\S4method{gamma}{Rcpp_MPCR}(x)

\S4method{lgamma}{Rcpp_MPCR}(x)
}
\arguments{
\item{x}{
An MPCR object.
}
}
\value{
An MPCR object of the same dimensions as the input.
}
\examples{
\dontrun{
library(MPCR)

x <- as.MPCR(1:20,precision="double")
lgamma(x)
}
}