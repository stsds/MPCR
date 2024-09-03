\name{32-Norm}
\alias{norm}
\alias{MPCR.norm}
\alias{norm,Rcpp_MPCR-method}
\title{norm}
\usage{
\S4method{norm}{Rcpp_MPCR}(x, type = "O")
}
\arguments{
\item{x}{An MPCR object.}

\item{type}{"O"-ne, "I"-nfinity, "F"-robenius, "M"-ax modulus, and "1" norms.}

}
\value{
An MPCR object.
}
\description{
Compute norm.
}
\examples{
library(MPCR)

x <- as.MPCR(1:20,precision="double")
norm(x, type="O")

}