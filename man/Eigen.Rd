\name{31-Eigen decomposition}
\alias{eigen}
\alias{eigen,Rcpp_MMPR-method}
\title{eigen decomposition}
\usage{
\S4method{eigen}{Rcpp_MMPR}(x, only.values = FALSE)
}
\arguments{
\item{x}{An MMPR object.}

\item{only.values}{(TRUE/FALSE)?}

}
\value{
A list contains MMPR objects describing the values and optionally vectors.
}
\description{
Solves a system of equations or invert an MMPR matrix, using lapack routine \code{syevr()}
}
\examples{
\dontrun{
library(MMPR)
x <- as.MMPR("your_data",nrow,ncol,precision)
y <- eigen(x)
}
}
