\name{sweep}
\alias{sweep}
\alias{sweep,Rcpp_MMPR-method}
\title{sweep}
\usage{
\S4method{sweep}{Rcpp_MMPR}(x,stat,margin,FUN)
}
\arguments{
\item{x}{An MMPR object.}

\item{stat}{MMPR vector containing the value(s) that should be used in the operation.}

\item{margin}{1 means row; otherwise means column.}

\item{FUN}{Sweeping function; must be one of \code{"+"}, \code{"-"}, \code{"*"}, \code{"/"}, or
\code{"^"}.}

}
\value{
An MMPR matrix of the same type as the highest precision input.
}
\description{
Sweep an MMPR vector through an MMPR matrix.
}
\examples{
\dontrun{
library(MMPR)
x <- as.MMPR(1:20,10,2,"single")
y <- as.MMPR(1:5,precision="double")
sweep_out <- sweep(x, stat=y, margin=1, FUN="+")
MMPR.is.double(sweep_out)  #TRUE
}
}
