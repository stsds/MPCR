\name{Converters}
\alias{Converters}
\title{Converters}


\section{MMPR Converter}{
    \description{
    Convert R Matrix/Vector to MMPR Matrix/Vector.
    }

   \subsection{MMPR Converters}{
     \code{as.MMPR(data,nrow = 0,ncol = 0,precision)}: Converts R matrix or vector to MMPR matrix or vector
      \cr
      \describe{
         \item{\code{data}}{R Matrix/Vector}
         \item{\code{nrow}}{Number of Rows of the new MMPR Matrix ,default = zero which means a vector will be created}
         \item{\code{ncol}}{Number of Cols of the new MMPR Matrix ,default = zero which means a vector will be created}
         \item{\code{precision}}{String indicating the precision of the new MMPR object (half, single, double).}
      }
   }
}

\section{R Converter}{
    \description{
    Convert MMPR object to R Matrix/Vector.
    }

   \subsection{R Vector Converter}{
     \code{MMPR.ToNumericVector(x)}: Converts MMPR object to Numeric R vector
      \describe{
         \item{\code{x}}{MMPR Matrix or Vector}
      }
   }

      \subsection{R Matrix Converter}{
        \code{MMPR.ToNumericMatrix(x)}: Converts MMPR object to Numeric R matrix
         \describe{
            \item{\code{x}}{MMPR Matrix or Vector}
         }
      }
}


\examples{
  \dontrun{
   # Example usage of the class and its methods
   a <- matrix(1:36, 6, 6)
   mmpr_matrix <- as.MMPR(a,nrow=6,ncol=6,precision="single")
   r_vector <- MMPR.ToNumericVector(mmpr_matrix)
   r_vector
   r_matrix <- MMPR.ToNumericMatrix(mmpr_matrix)
   r_matrix
  }
}