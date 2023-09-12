\name{10-Bind}

\alias{MPCR.rbind,Rcpp_MPCR-method}
\alias{MPCR.cbind,Rcpp_MPCR-method}
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
  \dontrun{
library(MPCR)
# create 2 MPCR matrix a,b
x <- rbind(a,b)
y <- cbind(a,b)
}
}
