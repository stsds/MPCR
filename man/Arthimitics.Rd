\name{04-Arithmetic}
\alias{arithmetic}

\alias{+,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{-,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{*,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{/,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{^,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{+,Rcpp_MPCR,BaseLinAlg-method}
\alias{*,Rcpp_MPCR,BaseLinAlg-method}
\alias{-,Rcpp_MPCR,BaseLinAlg-method}
\alias{/,Rcpp_MPCR,BaseLinAlg-method}
\alias{^,Rcpp_MPCR,BaseLinAlg-method}

\alias{MPCR.Add}
\alias{MPCR.Divide}
\alias{MPCR.Subtract}
\alias{MPCR.Multiply}
\alias{MPCR.Power}


\title{Binary arithmetic numeric/MPCR objects.}
\description{
Binary arithmetic for numeric/MPCR objects.
}

\usage{
\S4method{+}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{-}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{*}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{/}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{^}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{+}{Rcpp_MPCR,BaseLinAlg}(e1, e2)

\S4method{*}{Rcpp_MPCR,BaseLinAlg}(e1, e2)

\S4method{-}{Rcpp_MPCR,BaseLinAlg}(e1, e2)

\S4method{/}{Rcpp_MPCR,BaseLinAlg}(e1, e2)

\S4method{^}{Rcpp_MPCR,BaseLinAlg}(e1, e2)

}
\arguments{
\item{e1,e2}{
Numeric/MPCR objects.
}
}
\value{
An MPCR object, matching the data type of the highest precision input.
}
\examples{
library(MPCR)
s1 <- as.MPCR(1:20,nrow=2,ncol=10,"single")
s2 <- as.MPCR(21:40,nrow=2,ncol=10,"double")

x <- s1 + s2
typeof(x) # A 64-bit precision (double) MPCR matrix.

s3 <- as.MPCR(1:20,nrow=2,ncol=10,"single")
x <- s1 + s3
typeof(x) # A 32-bit precision (single) MPCR matrix.
}