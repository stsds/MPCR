\name{22-Trig}
\alias{trig}
\alias{sin,Rcpp_MPCR-method}
\alias{cos,Rcpp_MPCR-method}
\alias{tan,Rcpp_MPCR-method}
\alias{asin,Rcpp_MPCR-method}
\alias{acos,Rcpp_MPCR-method}
\alias{atan,Rcpp_MPCR-method}
\title{Trigonometric functions}
\description{
Basic trig functions.
}
\usage{
\S4method{sin}{Rcpp_MPCR}(x)

\S4method{cos}{Rcpp_MPCR}(x)

\S4method{tan}{Rcpp_MPCR}(x)

\S4method{asin}{Rcpp_MPCR}(x)

\S4method{acos}{Rcpp_MPCR}(x)

\S4method{atan}{Rcpp_MPCR}(x)
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

mpcr_matrix <- as.MPCR(1:20,nrow=2,ncol=10,"single")
x <- sin(mpcr_matrix)
}
}