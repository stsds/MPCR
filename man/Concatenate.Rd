\name{concatenate}
\alias{concatenate}
\alias{concatenate,Rcpp_MMPR-method}
\title{concatenate}
\usage{
\method{MMPR.Concatenate}{List of MMPR objects}(x)
}

\arguments{
\item{x}{List of MMPR matrix/vector(s).}



}
\value{
MMPR object containing all the values from all objects in the list.
}
\description{
\code{c()} for MMPR objects.
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
