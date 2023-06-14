\name{log}
\alias{log}
\alias{exp,Rcpp_MMPR-method}
\alias{expm1,Rcpp_MMPR-method}
\alias{log,Rcpp_MMPR-method}
\alias{log10,Rcpp_MMPR-method}
\alias{log2,Rcpp_MMPR-method}
\title{Logarithms and Exponentials}
\description{
exp/log functions.
}
\usage{
\S4method{exp}{Rcpp_MMPR}(x)

\S4method{expm1}{Rcpp_MMPR}(x)

\S4method{log}{Rcpp_MMPR}(x, base = 1)

\S4method{log10}{Rcpp_MMPR}(x)

\S4method{log2}{Rcpp_MMPR}(x)
}
\arguments{
\item{x}{
A MMPR vector/matrix.
}
\item{base}{
The logarithm base. if base = 1 , exp(1) is assumed , only base 1,2,10 available
}
}
\value{
A MMPR vector/matrix of the same dimensions as the input.
}
\examples{
\dontrun{
library(MMPR)

x <- as.MMPR(1:20,precision="double")
log(x)
}
}