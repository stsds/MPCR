\name{15-NA's}
\alias{NA's}
\alias{MPCR.is.na,Rcpp_MPCR-method}
\alias{MPCR.na.exclude,Rcpp_MPCR-method}
\alias{MPCR.na.omit,Rcpp_MPCR-method}
\title{NA's}
\usage{
\S4method{MPCR.is.na}{Rcpp_MPCR}(object,index=-1)
\S4method{MPCR.na.exclude}{Rcpp_MPCR}(object,value)
\S4method{MPCR.na.omit}{Rcpp_MPCR}(object)
}

\arguments{
\item{object}{MPCR object.}
\item{index}{If a particular index in the MPCR matrix/vector is specified, it will be checked. If no index is provided, all elements will be checked.}
\item{value}{Value to replace all NAN with.}


}
\value{
MPCR.is.na will return matrix/vector/bool according to input of the function.
\cr
MPCR.na.exclude & MPCR.na.omit will not return anything.
}
\description{
\code{is.na()} ,\code{na.omit()},  and \code{na.exclude()} for MPCR objects.
}

\examples{
  \donttest{
library(MPCR)
x <- as.MPCR(1:20,precision="single")
x[1] <- NAN
MPCR.is.na(x,index=1) #TRUE
MPCR.na.exclude(x,50)
x[1]  #50
}
}
