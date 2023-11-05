\name{05-Comparison}
\alias{comparison}

\alias{<,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{<=,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{==,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{!=,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{>,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{>=,Rcpp_MPCR,Rcpp_MPCR-method}
\alias{<,Rcpp_MPCR,BaseLinAlg-method}
\alias{<=,Rcpp_MPCR,BaseLinAlg-method}
\alias{==,Rcpp_MPCR,BaseLinAlg-method}
\alias{!=,Rcpp_MPCR,BaseLinAlg-method}
\alias{>,Rcpp_MPCR,BaseLinAlg-method}
\alias{>=,Rcpp_MPCR,BaseLinAlg-method}


\title{Binary comparison operators for numeric/MPCR objects.}
\description{
    Binary comparison operators for numeric/MPCR objects.
}
\usage{

\S4method{<}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{<=}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{==}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{!=}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{>}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{>=}{Rcpp_MPCR,Rcpp_MPCR}(e1, e2)

\S4method{<}{Rcpp_MPCR,BaseLinAlg}(e1, e2)

\S4method{<=}{Rcpp_MPCR,BaseLinAlg}(e1, e2)

\S4method{==}{Rcpp_MPCR,BaseLinAlg}(e1, e2)

\S4method{!=}{Rcpp_MPCR,BaseLinAlg}(e1, e2)

\S4method{>}{Rcpp_MPCR,BaseLinAlg}(e1, e2)

\S4method{>=}{Rcpp_MPCR,BaseLinAlg}(e1, e2)


}
\arguments{
  \item{e1,e2}{
    Numeric/MPCR objects.
  }
}
\value{
  A vector/matrix of logicals.
}
\examples{

  library(MPCR)
  s1 <- as.MPCR(1:20,nrow=2,ncol=10,"single")
  s2 <- as.MPCR(21:40,nrow=2,ncol=10,"double")

  x <- s1 > s2

}