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
   \donttest{
   library(MPCR)
   a <- matrix(c(3.12393, -1.16854, -0.304408, -2.15901,
                 -1.16854, 1.86968, 1.04094, 1.35925,
                 -0.304408, 1.04094, 4.43374, 1.21072,
                 -2.15901, 1.35925, 1.21072, 5.57265), 4,4)


   b <- c("float", "double", "float", "float")
   c <- c("float", "float", "double", "float")

   mat_a <- new(MPCRTile, 4,4, 2, 2, a, b)
   mat_b <- new(MPCRTile, 4,4, 2, 2, a, c)

   MPCRTile.trsm(a=mat_a,b=mat_b,side='R',upper_triangle=TRUE,transpose=FALSE,alpha=1)
   print(mat_b)

   }
}
