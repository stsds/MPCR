\name{09-Concatenate}
\alias{concatenate}
\alias{concatenate,Rcpp_MMPR-method}
\title{concatenate}
\usage{
\method{MMPR.Concatenate}{List of MMPR objects}(x)
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
