\name{09-Bind}

\alias{MPCR.rbind,Rcpp_MPCR-method}
\alias{MPCR.cbind,Rcpp_MPCR-method}

\alias{MPCR.rbind}
\alias{MPCR.cbind}

\title{bind}
\usage{
\S4method{MPCR.rbind}{Rcpp_MPCR}(x,y)

\S4method{MPCR.cbind}{Rcpp_MPCR}(x,y)
}
\arguments{
\item{x}{An MPCR object.}
\item{y}{An MPCR object.}


}
\value{
An MPCR object, matching the data type of the highest precision input.
}
\description{
\code{rbind()} and \code{cbind()} for MPCR objects.
}
\examples{
library(MPCR)
# create 2 MPCR matrix a,b
a <- as.MPCR(1:20,nrow=2,ncol=10,"single")
b <- as.MPCR(21:40,nrow=2,ncol=10,"double")

x <- MPCR.rbind(a,b)
y <- MPCR.cbind(a,b)

}
