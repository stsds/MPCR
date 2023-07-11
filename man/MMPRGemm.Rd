\name{MMPR GEMM}
\alias{MMPR GEMM}
\alias{MMPR GEMM,Rcpp_MMPR-method}
\title{MMPR GEMM  (Matrix-Matrix Multiplication)}

\description{
   Tiled-Matrix multiplication of 2 MMPR tile matrices.
   \bold{performs:}
   \cr
   C = alpha A * B + beta C
   \cr
   C = alpha A A^T + beta C
}
\usage{

\S4method{MMPR.gemm}{Rcpp_MMPR}(a,b = NULL,c,transpose_a= FALSE,transpose_b=FALSE,alpha=1,beta=0)
   }
 \arguments{
   \item{a}{MMPR tile matrix A.}
   \item{b}{MMPR tile matrix B, if NULL, the function will perform syrk operation from blas.}
   \item{c}{Input/Output MMPR matrix C.}
   \item{transpose_a}{Flag to indicate whether transpose matrix A should be used ,if B is NULL and transpose_a =TRUE
   \cr
   The function will perform the following operation :
   \cr
   \bold{C} = \bold{alpha A^T} X \bold{A} + \bold{beta C}.
   }
   \item{transpose_b}{Flag to indicate whether transpose matrix B should be used.}
   \item{alpha}{Specifies the scalar alpha.}
   \item{beta}{Specifies the scalar beta.}
 }
 \value{
   MMPR matrix C.
 }
 \examples{
   \dontrun{
   library(MMPR)
   # create 3 MMPR matrices a,b,c
   print(c)
   MMPR.gemm(a,b,c,transpose_a=false,transpose_b=TRUE,alpha=1,beta=1)
   print(c)
   }

