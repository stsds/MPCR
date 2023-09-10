\name{10-Bind}

\alias{MMPR.rbind,Rcpp_MMPR-method}
\alias{MMPR.cbind,Rcpp_MMPR-method}
\title{bind}
\usage{
\S4method{MMPR.rbind}{Rcpp_MMPR}(x,y)

\S4method{MMPR.cbind}{Rcpp_MMPR}(x,y)
}
\arguments{
\item{x}{An MMPR object.}
\item{y}{An MMPR object.}


}
\value{
An MMPR object, matching the data type of the highest precision input.
}
\description{
\code{rbind()} and \code{cbind()} for MMPR objects.
}
\examples{
  \dontrun{
library(MMPR)
# create 2 MMPR matrix a,b
x <- rbind(a,b)
y <- cbind(a,b)
}
}
