\name{06-Dimensions}
\alias{dimensions}
\alias{nrow,Rcpp_MPCR-method}
\alias{ncol,Rcpp_MPCR-method}
\alias{MPCR.ncol}
\alias{MPCR.nrow}
\title{dimensions}
\usage{
\S4method{nrow}{Rcpp_MPCR}(x)

\S4method{ncol}{Rcpp_MPCR}(x)

}
\arguments{
\item{x}{An MPCR object.}

}
\value{
The number of rows/cols in an MPCR object.
}
\description{
Returns the number of rows or cols in an MPCR object.
}
\examples{
    library(MPCR)
    x <- as.MPCR(1:16,4,4,"single")
    y <- as.MPCR(1:20,4,5,"double")
    rows_x <- nrow(x)
    cols_y <- ncol(y)
}
