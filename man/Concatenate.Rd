\name{08-Concatenate}
\alias{concatenate}
\alias{MPCR.Concatenate,Rcpp_MPCR-method}
\title{concatenate}
\usage{
\S4method{MPCR.Concatenate}{Rcpp_MPCR}(x)
}


\arguments{
\item{x}{List of MPCR objects.}
}
\value{
MPCR object containing values from all objects in the list.
}
\description{
\code{c()} function for MPCR objects.
}

\examples{
library(MPCR)
x <- as.MPCR(1:20,precision="single")
y <- as.MPCR(1:20,precision="single")
list <- c(x,y)
new_obj <- MPCR.Concatenate(list)
}
