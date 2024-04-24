\name{35-Reciprocal condition}
\alias{rcond}
\alias{rcond,Rcpp_MPCR-method}
\alias{MPCR.rcond}
\title{reciprocal condition}
\usage{
\S4method{rcond}{Rcpp_MPCR}(x, norm = "O", useInv = FALSE)
}
\arguments{
\item{x}{An MPCR object.}

\item{norm}{"O"-ne or "I"-nfinity norm.}

\item{useInv}{TRUE to use the lower triangle only.}

}
\value{
An MPCR Object.
}
\description{
Compute matrix norm.
}
\examples{
\donttest{
library(MPCR)

x <- as.MPCR(1:25,precision="double")
x$ToMatrix(5,5)

rcond(x)
}
}