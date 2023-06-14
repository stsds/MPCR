\name{NA's}
\alias{NA's}
\alias{is.na,Rcpp_MMPR-method}
\alias{na.exclude,Rcpp_MMPR-method}
\alias{na.omit,Rcpp_MMPR-method}
\title{NA's}
\usage{
\method{MMPR.is.na}{Rcpp_MMPR}(object,index=-1)
\method{MMPR.na.exclude}{Rcpp_MMPR}(object,value)
\method{MMPR.na.omit}{Rcpp_MMPR}(object)
}

\arguments{
\item{object}{MMPR matrix/vector.}
\item{index}{specific index in the MMPR matrix/vector to check. if index is not provided all the elements will be checked}
\item{value}{value to replace all NAN with.}


}
\value{
MMPR.is.na will return matrix/vector/bool according to input of the function
\cr
MMPR.na.exclude & MMPR.na.omit will not return anything.
}
\description{
\code{is.na()} ,\code{na.omit()},  and \code{na.exclude()} for MMPR objects.
}

\examples{
  \dontrun{
library(MMPR)
x <- as.MMPR(1:20,precision"single")
x[1] <- NAN
MMPR.is.na(x,index=1) #TRUE
MMPR.na.exclude(x,50)
x[1]  #50
}
}
