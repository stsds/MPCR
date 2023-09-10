\name{09-Concatenate}
\alias{concatenate}
\alias{MMPR.Concatenate,Rcpp_MMPR-method}
\title{concatenate}
\usage{
\S4method{MMPR.Concatenate}{Rcpp_MMPR}(x)
}


\arguments{
\item{x}{List of MMPR objects.}
}
\value{
MMPR object containing values from all objects in the list.
}
\description{
\code{c()} function for MMPR objects.
}

\examples{
  \dontrun{
library(MMPR)
x <- as.MMPR(1:20,precision"single")
y <- as.MMPR(1:20,precision"single")
list <- c(x,y)
new_obj <- MMPR.Concatenate(list)
}
}
