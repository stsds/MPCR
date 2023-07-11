\name{rcond}
\alias{rcond}
\alias{rcond,Rcpp_MMPR-method}
\title{rcond}
\usage{
\S4method{rcond}{Rcpp_MMPR}(x, norm = "O", useInv = FALSE)
}
\arguments{
\item{x}{An MMPR object.}

\item{norm}{"O"-ne or "I"-nfinity norm.}

\item{useInv}{TRUE to use the lower triangle only.}

}
\value{
An MMPR Object.
}
\description{
Compute matrix norm.
}
\examples{
\dontrun{
library(MMPR)

x <- as.MMPR(1:20,precision="double")
rcond(x)
}
}