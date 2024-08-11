\name{14-Miscmath}
\alias{miscmath}
\alias{abs,Rcpp_MPCR-method}
\alias{sqrt,Rcpp_MPCR-method}

\alias{MPCR.abs}
\alias{MPCR.sqrt}

\title{Miscellaneous mathematical functions}
\description{
Miscellaneous mathematical functions.
}
\usage{
\S4method{abs}{Rcpp_MPCR}(x)

\S4method{sqrt}{Rcpp_MPCR}(x)
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
\donttest{
library(MPCR)

x <- as.MPCR(1:20,precision="double")
sqrt(x)
}
}