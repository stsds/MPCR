\name{MMPRTile GEMM}
\alias{MMPRTile GEMM}
\alias{MMPRTile GEMM,Rcpp_MMPRTile-method}
\title{MMPRTile GEMM  (Matrix-Matrix Multiplication)}

\description{
   Tiled-Matrix multiplication of 2 MMPR tile matrices.
   performs:
   C = \bold{alpha*A} X \bold{B} + \bold{beta*C}
}
\usage{

\S4method{MMPRTile.gemm}{Rcpp_MMPRTile}(a,b,c,transpose_a= FALSE,transpose_b=FALSE,alpha=1,beta=0)
   }
 \arguments{
   \item{a}{MMPR tile matrix A.}
   \item{b}{MMPR tile matrix B.}
   \item{c}{Input/Output MMPR tile matrix C.}
   \item{transpose_a}{Flag to indicate whether transpose matrix A should be used.}
   \item{transpose_b}{Flag to indicate whether transpose matrix B should be used.}
   \item{alpha}{Specifies the scalar alpha.}
   \item{beta}{Specifies the scalar beta.}
    \item{num_threads}{
       Integer to determine number if thread to run using openmp, \bold{default = 1} (serial with no parallelization).
     }
 }
 \value{
   MMPR tile matrix C.
 }
 \examples{
   \dontrun{
   library(MMPR)
   # create 3 MMPR Tile matrices a,b,c
   print(c)
   MMPRTile.gemm(a,b,c,transpose_a=false,transpose_b=TRUE,alpha=1,beta=1,num_threads = 8)
   print(c)
   }

