\name{12-Log}
\alias{log}
\alias{exp,Rcpp_MPCR-method}
\alias{expm1,Rcpp_MPCR-method}
\alias{log,Rcpp_MPCR-method}
\alias{log10,Rcpp_MPCR-method}
\alias{log2,Rcpp_MPCR-method}
\title{Logarithms and Exponentials}
\description{
exp/log functions.
}
\usage{
\S4method{exp}{Rcpp_MPCR}(x)

\S4method{expm1}{Rcpp_MPCR}(x)

\S4method{log}{Rcpp_MPCR}(x, base = 1)

\S4method{log10}{Rcpp_MPCR}(x)

\S4method{log2}{Rcpp_MPCR}(x)
}
\arguments{
\item{x}{
An MPCR object.
}
\item{base}{
The logarithm base. If base = 1, exp(1) is assumed, only base 1,2, and 10 available.
}
}
\value{
An MPCR object of the same dimensions as the input.
}
\examples{
\donttest{
library(MPCR)

x <- as.MPCR(1:20,precision="double")
log(x)
}
}