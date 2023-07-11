\name{Converters}
\alias{Converters}
\title{Converters}


\section{MMPR Converter}{
    \description{
    Convert R object to MMPR object.
    }

   \subsection{MMPR converters}{
   \cr
     \code{as.MMPR(data,nrow = 0,ncol = 0,precision)}: Converts R object to MMPR object.
      \cr
      \describe{
         \item{\code{data}}{R matrix/vector.}
         \item{\code{nrow}}{Number of rows of the new MMPR matrix, \bold{default = zero} which means a vector will be created.}
         \item{\code{ncol}}{Number of cols of the new MMPR matrix, \bold{default = zero} which means a vector will be created.}
         \item{\code{precision}}{String indicating the precision of the new MMPR object (half, single, or double).}
      }
   }
}

\section{R Converter}{
    \description{
    Convert MMPR object to R object.
    }

   \subsection{R vector converter}{
   \cr
     \code{MMPR.ToNumericVector(x)}: Converts MMPR object to numeric R vector.
      \describe{
         \item{\code{x}}{MMPR object.}
      }
   }

      \subsection{R matrix converter}{
      \cr
        \code{MMPR.ToNumericMatrix(x)}: Converts MMPR object to numeric R matrix.
         \describe{
            \item{\code{x}}{MMPR object.}
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