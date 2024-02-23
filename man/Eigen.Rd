\name{31-Eigen decomposition}
\alias{eigen}
\alias{MPCR.eigen}
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
