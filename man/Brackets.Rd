\name{06-Extract-Replace}
\alias{[,Rcpp_MPCR-method}
\alias{[[,Rcpp_MPCR-method}
\alias{[<-,Rcpp_MPCR-method}
\alias{[[<-,Rcpp_MPCR-method}
\title{Extract or replace elements from an MPCR object.}
\usage{
  \S4method{[}{Rcpp_MPCR}(x, i, j, drop = TRUE)
  \S4method{[}{Rcpp_MPCR}(x, i, j, ...) <- value
  \S4method{[[}{Rcpp_MPCR}(x, i, drop = TRUE)
  \S4method{[[}{Rcpp_MPCR}(x, i, ...) <- value
}
\arguments{
  \item{x}{An MPCR object.}
  \item{i}{Row index or indices.}
  \item{j}{Column index or indices.}
  \item{...}{ignored.}
  \item{drop}{ignored.}
  \item{value}{A value to replace the selected elements with.}
}
\description{
Extract or replace elements from an MPCR object using the `[`, `[[`, `[<-`, and `[[<-` operators.
When extracting values, they will be converted to double precision. However, if you update a single object, the double value will be cast down to match the precision.
If the MPCR object is a matrix and you access it using the 'i' index, the operation is assumed to be performed in column-major order, or using 'i' and 'j' index.
}
\examples{
  library(MPCR)
    x <-as.MPCR(1:50,precision="single")
    ext <- x[5]
    x[5] <- 0
    x$ToMatrix(5,10)
    x[2,5]
    x[3,5] <- 100

}
