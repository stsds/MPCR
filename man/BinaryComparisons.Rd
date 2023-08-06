\name{Comparison}
\alias{comparison}
\title{Binary comparison operators for numeric/MMPR objects.}
\description{
    Binary comparison operators for numeric/MMPR objects.
}
\usage{

\S4method{<}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{<=}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{==}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{!=}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{>}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{>=}{Rcpp_MMPR,Rcpp_MMPR}(e1, e2)

\S4method{<}{Rcpp_MMPR,BaseLinAlg}(e1, e2)

\S4method{<=}{Rcpp_MMPR,BaseLinAlg}(e1, e2)

\S4method{==}{Rcpp_MMPR,BaseLinAlg}(e1, e2)

\S4method{!=}{Rcpp_MMPR,BaseLinAlg}(e1, e2)

\S4method{>}{Rcpp_MMPR,BaseLinAlg}(e1, e2)

\S4method{>=}{Rcpp_MMPR,BaseLinAlg}(e1, e2)


}
\arguments{
  \item{e1,e2}{
    Numeric/MMPR objects.
  }
}
\value{
  A vector/matrix of logicals.
}
\examples{
  \dontrun{
  library(MMPR)
  s1 <- as.MMPR(1:20,nrow=2,ncol=10,"single")
  s2 <- as.MMPR(21:40,nrow=2,ncol=10,"double")

  x <- s1 > s2
  }
}