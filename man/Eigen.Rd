\name{eigen}
\alias{eigen}
\alias{eigen,Rcpp_MMPR-method}
\title{eigen}
\usage{
\S4method{eigen}{Rcpp_MMPR}(x, only.values = FALSE)
}
\arguments{
\item{x}{A MMPR vector/matrix.}

\item{only.values}{Should only the values (and not the vectors) be returned?}

}
\value{
A list containing MMPR objects describing the values and optionally vectors.
}
\description{
Solve a system of equations or invert a MMPR matrix, using lapack routine \code{syevr()}
}
\examples{
\dontrun{
library(MMPR)
x <- as.MMPR("your_data",nrow,ncol,precision)
y <- eigen(x)
}
}
