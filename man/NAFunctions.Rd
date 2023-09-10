\name{16-NA's}
\alias{NA's}
\alias{MMPR.is.na,Rcpp_MMPR-method}
\alias{MMPR.na.exclude,Rcpp_MMPR-method}
\alias{MMPR.na.omit,Rcpp_MMPR-method}
\title{NA's}
\usage{
\S4method{MMPR.is.na}{Rcpp_MMPR}(object,index=-1)
\S4method{MMPR.na.exclude}{Rcpp_MMPR}(object,value)
\S4method{MMPR.na.omit}{Rcpp_MMPR}(object)
}

\arguments{
\item{object}{MMPR object.}
\item{index}{If a particular index in the MMPR matrix/vector is specified, it will be checked. If no index is provided, all elements will be checked.}
\item{value}{Value to replace all NAN with.}


}
\value{
MMPR.is.na will return matrix/vector/bool according to input of the function.
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
