\name{MMPRTile TRSM}
\alias{MMPRTile TRSM}
\alias{MMPRTile TRSM,Rcpp_MMPRTile-method}
\title{MMPRTile TRSM  (Triangular Solve)}

\description{
            Solves a triangular matrix equation performs:
            \cr
            op(A)*X=alpha*B
            \cr
            X*op(A)=alpha*B
}
\usage{

\S4method{MMPRTile.trsm}{Rcpp_MMPRTile}(a,b,side,upper_triangle,transpose,alpha)
   }
 \arguments{
   \item{a}{MMPR Tile Matrix A}
   \item{b}{MMPRTile Matrix B, X after returning.}
   \item{side}{'R for Right side , 'L' for Left side}
   \item{upper_triangle}{What part of the matrix A is referenced (if TRUE upper triangle is referenced), the opposite triangle
    being assumed to be zero.}
   \item{transpose}{if TRUE , the transpose of A is used.}
   \item{alpha}{Factor used for A , If alpha is zero, A is not accessed.}
 }
 \value{
   MMPR Tile Matrix B ->(X)
 }
 \examples{
   \dontrun{
   library(MMPR)
   # create 2 MMPR Tile matrices a,b
   print(b)
   MMPRTile.trsm(a,b,side='L',upper_triangle=TRUE,transpose=FALSE,alpha=1)
   print(b)
   }

