\name{MMPRTile GEMM}
\alias{MMPRTile GEMM}
\alias{MMPRTile GEMM,Rcpp_MMPRTile-method}
\title{MMPRTile GEMM  (Matrix-Matrix Multiplication)}

\description{
   Tiled-Matrix Multiplication of 2 MPR Tile Matrices
   performs:
   C = alpha A * B + beta C
}
\usage{

\S4method{MMPRTile.gemm}{Rcpp_MMPRTile}(a,b,c,transpose_a= FALSE,transpose_b=FALSE,alpha=1,beta=0)
   }
 \arguments{
   \item{a}{MMPR Tile Matrix A}
   \item{b}{MMPR Tile Matrix B}
   \item{c}{Input/Output MMPR Tile Matrix C}
   \item{transpose_a}{Flag to indicate whether transpose matrix A should be used}
   \item{transpose_b}{Flag to indicate whether transpose matrix B should be used}
   \item{alpha}{Specifies the scalar alpha.}
   \item{beta}{Specifies the scalar beta.}
 }
 \value{
   MMPR Tile Matrix C
 }
 \examples{
   \dontrun{
   library(MMPR)
   # create 3 MMPR Tile matrices a,b,c
   print(c)
   MMPRTile.gemm(a,b,c,transpose_a=false,transpose_b=TRUE,alpha=1,beta=1)
   print(c)
   }

