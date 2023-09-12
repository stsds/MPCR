\name{40-MPCR TRSM}
\alias{MPCR TRSM}
\alias{MPCR.trsm,Rcpp_MPCR-method}
\title{MPCR TRSM  (Triangular Solve)}

\description{
            Solves a triangular matrix equation.
             \cr
            performs:
            \cr
            op(A)*X=alpha*B
            \cr
            X*op(A)=alpha*B
}
\usage{

\S4method{MPCR.trsm}{Rcpp_MPCR}(a,b,upper_triangle,transpose,side = 'L',alpha =1)
   }
 \arguments{
   \item{a}{MPCR Matrix A.}
   \item{b}{MPCR Matrix B.}
   \item{upper_triangle}{If the value is TRUE, the referenced part of matrix A corresponds to the upper triangle, with the opposite triangle assumed to contain zeros.}
   \item{transpose}{If TRUE, the transpose of A is used.}
   \item{side}{'R for Right side, 'L' for Left side.}
   \item{alpha}{Factor used for A, If alpha is zero, A is not accessed.}
 }
 \value{
   An MPCR Matrix.
 }
 \examples{
   \dontrun{
   library(MPCR)
   # create 2 MPCR matrices a,b
   c <- MPCR.trsm(a,b,upper_triangle=TRUE,transpose=FALSE,side='L',alpha=1)
   print(c)
   }
}
