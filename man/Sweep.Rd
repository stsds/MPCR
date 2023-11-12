\name{19-Sweep}
\alias{sweep}
\alias{sweep,Rcpp_MPCR-method}
\title{sweep}
\usage{
\S4method{sweep}{Rcpp_MPCR}(x,stat,margin,FUN)
}
\arguments{
\item{x}{An MPCR object.}

\item{stat}{MPCR vector containing the value(s) that should be used in the operation.}

\item{margin}{1 means row; otherwise means column.}

\item{FUN}{Sweeping function; must be one of \code{"+"}, \code{"-"}, \code{"*"}, \code{"/"}, or
\code{"^"}.}

}
\value{
An MPCR matrix of the same type as the highest precision input.
}
\description{
Sweep an MPCR vector through an MPCR matrix.
}
\examples{
\donttest{
library(MPCR)
x <- as.MPCR(1:20,10,2,"single")
y <- as.MPCR(1:5,precision="double")
sweep_out <- sweep(x, stat=y, margin=1, FUN="+")
MPCR.is.double(sweep_out)  #TRUE
}
}
