\name{norm}
\alias{norm}
\alias{norm,Rcpp_MMPR,ANY-method}
\title{norm}
\usage{
\S4method{norm}{Rcpp_MMPR}(x, type = "O")
}
\arguments{
\item{x}{An MMPR object.}

\item{type}{"O"-ne, "I"-nfinity, "F"-robenius, "M"-ax modulus, and "1" norms.}

}
\value{
An MMPR object.
}
\description{
Compute matrix norm.
}
\examples{
\dontrun{
library(MMPR)

x <- as.MMPR(1:20,precision="double")
norm(x, type="O")
}
}