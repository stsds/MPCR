\name{extract-replace}
\alias{extract-replace}
\alias{[,Rcpp_MMPR}
\alias{[[,Rcpp_MMPR}
\alias{[<-,Rcpp_MMPR}
\alias{[[<-,Rcpp_MMPR}
\title{Extract or replace elements from an Rcpp_MMPR object}
\usage{
  \method{[}{Rcpp_MMPR}(x, i, j, drop = TRUE)
  \method{[<-}{Rcpp_MMPR}(x, i, j, ..., value)
  \method{[[}{Rcpp_MMPR}(x, i, drop = TRUE)
  \method{[[<-}{Rcpp_MMPR}(x, i, ..., value)
}
\arguments{
  \item{x}{An MMPR object vector/matrix.}
  \item{i}{Row index or indices.}
  \item{j}{Column index or indices.}
  \item{drop}{ignored}
  \item{value}{A value to replace the selected elements with.}
}
\description{
Extract or replace elements from an MMPR object using the `[`, `[[`, `[<-`, and `[[<-` operators.
all extracted values will be double precision, however, when you need to update a single object , the double
value will be casted down to match the precision.
Incase the MMPR object is a Matrix and the accessing is done with only i, column major is assumed for the operation.
}
\examples{
  \dontrun{
  library(MMPR)
    x <-as.MMPR(1:50,"single")
    ext <- x[5]
    x[5] <- 0
    x$ToMatrix(5,10)
    x[2,5]
    x[3,5] <- 100

}

}
