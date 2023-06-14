\name{MMPR TRSM}
\alias{MMPR TRSM}
\alias{MMPR TRSM,Rcpp_MMPR-method}
\title{MMPR TRSM  (Triangular Solve)}

\description{
            Solves a triangular matrix equation performs:
            \cr
            op(A)*X=alpha*B
            \cr
            X*op(A)=alpha*B
}
\usage{

\S4method{MMPR.trsm}{Rcpp_MMPR}(a,b,upper_triangle,transpose,side = 'L',alpha =1)
   }
 \arguments{
   \item{a}{MMPR Matrix A}
   \item{b}{MMPR Matrix B}
   \item{upper_triangle}{What part of the matrix A is referenced (if TRUE upper triangle is referenced), the opposite triangle
    being assumed to be zero.}
   \item{transpose}{if TRUE , the transpose of A is used.}
   \item{side}{'R for Right side , 'L' for Left side}
   \item{alpha}{Factor used for A , If alpha is zero, A is not accessed.}
 }
 \value{
   MMPR Matrix
 }
 \examples{
   \dontrun{
   library(MMPR)
   # create 2 MMPR matrices a,b
   c <- MMPR.trsm(a,b,upper_triangle=TRUE,transpose=FALSE,side='L',alpha=1)
   print(c)
   }

