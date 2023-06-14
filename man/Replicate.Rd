\name{replicate}
\alias{replicate}
\alias{rep,Rcpp_MMPR-method}
\title{replicate}
\usage{
\S4method{rep}{Rcpp_MMPR}(x,count=0,len=0)
}
\arguments{
\item{x}{A MMPR vector/matrix.}
\item{count}{value to determine how many times the input value will be replicated}
\item{len}{value to determine the required output size, the input
will be replicated until it matches the output len size}

}
\value{
MMPR vector containing the replicated values
}
\description{
Replicate the given input number of times according to count/len , only one should be set at a time, and incase both
values are given, only the len value will have effect.
}
\examples{
  \dontrun{
    library(MMPR)
    x <- as.MMPR(1:16,4,4,"single")
    rep_vals_1 <- rep(x,count=2)  #output size will be 16*2
    rep_vals_2 <- rep(x,len=2)  #output size will be 2
    }

}
