\name{38-MPCR GEMM}
\alias{MPCR GEMM}
\alias{MPCR.gemm}
\alias{MPCR.gemm,Rcpp_MPCR-method}
\title{MPCR GEMM  (Matrix-Matrix Multiplication)}

\description{
   Performs matrix-matrix multiplication of two given MPCR matrices to performs:
   \cr
   C = alpha A * B + beta C
   \cr
   C = alpha A A^T + beta C
}
\usage{

\S4method{MPCR.gemm}{Rcpp_MPCR}(a,b = NULL,c,transpose_a= FALSE,transpose_b=FALSE,alpha=1,beta=0)
   }
 \arguments{
   \item{a}{An MPCR matrix A.}
   \item{b}{An MPCR matrix B, if NULL, the function will perform syrk operation from blas.}
   \item{c}{Input/Output MPCR matrix C.}
   \item{transpose_a}{A flag to indicate whether transpose matrix A should be used, if B is NULL and transpose_a =TRUE
   \cr
   The function will perform the following operation:
   \cr
   \bold{C}=\bold{alphaA^T}X\bold{A}+\bold{betaC}.
   }
   \item{transpose_b}{A flag to indicate whether transpose matrix B should be used.}
   \item{alpha}{Specifies the scalar alpha.}
   \item{beta}{Specifies the scalar beta.}
 }
 \value{
   An MPCR matrix.
 }
 \examples{
   \dontrun{
   library(MPCR)
   # create 3 MPCR matrices a,b,c
   a <- as.MPCR(1:12, 3, 4, "double")
   b <- as.MPCR(1:20, 4, 5, "double")
   c <- as.MPCR(matrix(0, 3, 5), 3, 5, "double")
   print(c)
   MPCR.gemm(a,b,c,transpose_a=FALSE,transpose_b=TRUE,alpha=1,beta=1)
   print(c)
   }
 }

