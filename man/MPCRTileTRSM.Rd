\name{43-MPCRTile TRSM}


\alias{MPCRTile.trsm,Rcpp_MPCRTile-method}
\title{MPCRTile TRSM  (Triangular Solve)}

\description{
            Tile-based algorithm to solve a triangular matrix equation for MPCR tiled matrices.
            \cr
            performs:
            \cr
            op(A)*X=alpha*B
            \cr
            X*op(A)=alpha*B
}
\usage{

\S4method{MPCRTile.trsm}{Rcpp_MPCRTile}(a,b,side,upper_triangle,transpose,alpha)
   }
 \arguments{
   \item{a}{An MPCR tile matrix A.}
   \item{b}{An MPCR tile matrix B, X after returning.}
   \item{side}{'R' for right side, 'L' for left side.}
   \item{upper_triangle}{What part of the matrix A is referenced (if TRUE upper triangle is referenced), the opposite triangle
    being assumed to be zero.}
   \item{transpose}{If TRUE, the transpose of A is used.}
   \item{alpha}{Factor used for A, If alpha is zero, A is not accessed.}
 }
 \value{
   An MPCR Tile Matrix B ->(X).
 }

 \examples{
   \dontrun{
   library(MPCR)
   # create 2 MPCR Tile matrices a,b
   print(b)
   MPCRTile.trsm(a,b,side='L',upper_triangle=TRUE,transpose=FALSE,alpha=1)
   print(b)
   }
}
