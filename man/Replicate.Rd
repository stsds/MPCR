\name{17-Replicate}
\alias{replicate}
\alias{MPCR.rep}
\alias{rep,Rcpp_MPCR-method}
\title{replicate}
\usage{
\S4method{rep}{Rcpp_MPCR}(x,count=0,len=0)
}
\arguments{
\item{x}{An MPCR object.}
\item{count}{Value to determine how many times the input value will be replicated.}
\item{len}{Value to determine the required output size, the input
will be replicated until it matches the output len size.}

}
\value{
MPCR vector containing the replicated values.
}
\description{
Replicates the given input number of times according to count/len , only one should be set at a time, and in case both
values are given, only the len value will have effect.
}
\examples{

    library(MPCR)
    x <- as.MPCR(1:16,4,4,"single")
    rep_vals_1 <- rep(x,count=2)  #output size will be 16*2
    rep_vals_2 <- rep(x,len=2)  #output size will be 2


}
