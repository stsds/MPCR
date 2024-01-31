\name{41-MPCRTile GEMM}
\alias{MPCRTile GEMM}
\alias{MPCRTile.gemm}
\alias{MPCRTile.gemm,Rcpp_MPCRTile-method}
\title{MPCRTile GEMM  (Matrix-Matrix Multiplication)}

\description{
   Tile-based matrix-matrix multiplication of two given MPCR tiled matrices to
   \bold{perform:}
   \cr
   C = \bold{alpha*A} X \bold{B} + \bold{beta*C}
}
\usage{

\S4method{MPCRTile.gemm}{Rcpp_MPCRTile}(a,b,c,transpose_a= FALSE,transpose_b=FALSE,alpha=1,beta=0,num_threads=1)
   }
 \arguments{
   \item{a}{An MPCR tile matrix A.}
   \item{b}{An MPCR tile matrix B.}
   \item{c}{Input/Output MPCR tile matrix C.}
   \item{transpose_a}{A flag to indicate whether transpose matrix A should be used.}
   \item{transpose_b}{A flag to indicate whether transpose matrix B should be used.}
   \item{alpha}{Specifies the scalar alpha.}
   \item{beta}{Specifies the scalar beta.}
    \item{num_threads}{
      An integer to determine number if thread to run using openmp, default = 1 (serial with no parallelization).
     }
 }
 \value{
   An MPCR tile matrix C.
 }
