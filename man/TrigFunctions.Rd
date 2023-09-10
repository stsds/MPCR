\name{22-Trig}
\alias{trig}
\alias{sin,Rcpp_MMPR-method}
\alias{cos,Rcpp_MMPR-method}
\alias{tan,Rcpp_MMPR-method}
\alias{asin,Rcpp_MMPR-method}
\alias{acos,Rcpp_MMPR-method}
\alias{atan,Rcpp_MMPR-method}
\title{Trigonometric functions}
\description{
Basic trig functions.
}
\usage{
\S4method{sin}{Rcpp_MMPR}(x)

\S4method{cos}{Rcpp_MMPR}(x)

\S4method{tan}{Rcpp_MMPR}(x)

\S4method{asin}{Rcpp_MMPR}(x)

\S4method{acos}{Rcpp_MMPR}(x)

\S4method{atan}{Rcpp_MMPR}(x)
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

mppr_matrix <- as.MMPR(1:20,nrow=2,ncol=10,"single")
x <- sin(mppr_matrix)
}
}