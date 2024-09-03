\name{39-MPCR TRSM}
\alias{MPCR TRSM}
\alias{MPCR.trsm}
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
   \donttest{
   library(MPCR)
   a <- matrix(c(3.12393, -1.16854, -0.304408, -2.15901,
                 -1.16854, 1.86968, 1.04094, 1.35925,
                 -0.304408, 1.04094, 4.43374, 1.21072,
                 -2.15901, 1.35925, 1.21072, 5.57265), 4,4)


   mat_a <- as.MPCR(a,4,4,"single")
   mat_b <- as.MPCR(a,4,4,"double")

   MPCR.trsm(a=mat_a,b=mat_b,side='R',upper_triangle=TRUE,transpose=FALSE,alpha=1)
   print(mat_b)
   }
}
