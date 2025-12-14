\name{34-Reciprocal condition}
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

m <- matrix(c(3.12393, -1.16854, -0.304408, -2.15901,
                 -1.16854, 1.86968, 1.04094, 1.35925,
                 -0.304408, 1.04094, 4.43374, 1.21072,
                 -2.15901, 1.35925, 1.21072, 5.57265), 4,4)

x <- as.MPCR(m,4,4,"double")

rcond(x)
}
}
