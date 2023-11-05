\name{31-Eigen decomposition}
\alias{eigen}
\alias{eigen,Rcpp_MPCR-method}
\title{eigen decomposition}
\usage{
\S4method{eigen}{Rcpp_MPCR}(x, only.values = FALSE)
}
\arguments{
\item{x}{An MPCR object.}

\item{only.values}{(TRUE/FALSE)?}

}
\value{
A list contains MPCR objects describing the values and optionally vectors.
}
\description{
Solves a system of equations or invert an MPCR matrix, using lapack routine \code{syevr()}
}
\examples{
\donttest{
library(MPCR)
s <- runif(10, 3)
cross_prod <- crossprod(s)
x <- as.MPCR(cross_prod,nrow(cross_prod),nrow(cross_prod),precision)
y <- eigen(x)
}
}
