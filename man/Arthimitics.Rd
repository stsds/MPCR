\name{arithmetic}
\alias{arithmetic}
\title{Binary arithmetic numeric/MMPR objects.}
\description{
Binary arithmetic for numeric/MMPR objects.
}
\usage{
\S4method{+}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{-}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{*}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{/}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{^}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{+}{Rcpp_MMPR,BaseLinAlg}(e1, e2)

\S4method{*}{Rcpp_MMPR,BaseLinAlg}(e1, e2)

\S4method{-}{Rcpp_MMPR,BaseLinAlg}(e1, e2)

\S4method{/}{Rcpp_MMPR,BaseLinAlg}(e1, e2)

\S4method{^}{Rcpp_MMPR,BaseLinAlg}(e1, e2)

}
\arguments{
\item{e1,e2}{
Numeric/MMPR objects.
}
}
\value{
An MMPR object, matching the data type of the highest precision input.
}
\examples{
\dontrun{
library(MMPR)
s1 <- as.MMPR(1:20,nrow=2,ncol=10,"single")
s2 <- as.MMPR(21:40,nrow=2,ncol=10,"double")

x <- s1 + s2
typeof(x) # A 64-bit precision (double) MMPR matrix.

s3 <- as.MMPR(1:20,nrow=2,ncol=10,"single")
x <- s1 + s3
typeof(x) # A 32-bit precision (single) MMPR matrix.
}
}