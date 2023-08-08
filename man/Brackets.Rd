\name{06-Extract-Replace}
\alias{extract-replace}
\alias{[,Rcpp_MMPR}
\alias{[[,Rcpp_MMPR}
\alias{[<-,Rcpp_MMPR}
\alias{[[<-,Rcpp_MMPR}
\title{Extract or replace elements from an MMPR object.}
\usage{
  \method{[}{Rcpp_MMPR}(x, i, j, drop = TRUE)
  \method{[<-}{Rcpp_MMPR}(x, i, j, ..., value)
  \method{[[}{Rcpp_MMPR}(x, i, drop = TRUE)
  \method{[[<-}{Rcpp_MMPR}(x, i, ..., value)
}
\arguments{
  \item{x}{An MMPR object.}
  \item{i}{Row index or indices.}
  \item{j}{Column index or indices.}
  \item{drop}{ignored.}
  \item{value}{A value to replace the selected elements with.}
}
\description{
Extract or replace elements from an MMPR object using the `[`, `[[`, `[<-`, and `[[<-` operators.
When extracting values, they will be converted to double precision. However, if you update a single object, the double value will be cast down to match the precision.
If the MMPR object is a matrix and you access it using the 'i' index, the operation is assumed to be performed in column-major order, or using 'i' and 'j' index.
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
