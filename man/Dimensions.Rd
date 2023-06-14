\name{dimensions}
\alias{dimensions}
\alias{nrow,Rcpp_MMPR-method}
\alias{ncol,Rcpp_MMPR-method}
\title{dimensions}
\usage{
\S4method{nrow}{Rcpp_MMPR}(x)

\S4method{ncol}{Rcpp_MMPR}(x)

}
\arguments{
\item{x}{A MMPR vector/matrix.}

}
\value{
The number of rows/cols in MMPR vector/matrix
}
\description{
Dimension information for a MMPR vector/matrix.
}
\examples{
  \dontrun{
    library(MMPR)
    x <- as.MMPR(1:16,4,4,"single")
    y <- as.MMPR(1:20,4,5,"double")
    rows_x <- nrow(x)
    cols_y <- ncol(y)
    }

}
