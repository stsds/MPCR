\name{bind}
\alias{bind}
\alias{rbind.Rcpp_MMPR-method}
\alias{cbind.Rcpp_MMPR-method}
\title{rbind}
\usage{
\method{MMPR.rbind}{Rcpp_MMPR}(x,y)

\method{MMPR.cbind}{Rcpp_MMPR}(x,y)
}
\arguments{
\item{x}{MMPR matrix/vector}
\item{y}{MMPR matrix/vector}


}
\value{
A MMPR matrix of the same type as the highest precision input.
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
