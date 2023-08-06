\name{Copy}
\alias{copy}
\title{copy}


\section{MMPR deep copy}{
    \description{
   Create a copy of an MMPR object. Typically, using 'equal' creates a new pointer for the object, resulting in any modifications made to object one affecting object two as well.
    }

    \subsection{copy}{
    \cr
      \code{MMPR.copy(x)}: Create a new copy of an MMPR object.
      \describe{
      \item{\code{x}}{MMPR object. }
    }}

}

\section{MMPRTile deep copy}{
    \description{
    Create a duplicate of an MMPRTile object. Usually, using 'equal' creates a new pointer for the object, causing any modifications made to object one to affect object two as well.
    }

   \subsection{copy}{
   \cr
     \code{MMPRTile.copy(x)}: Create a new copy of an MMPRTile matrix.
      \describe{
         \item{\code{x}}{MMPRTile matrix. }
      }
   }
}


\examples{
  \dontrun{
   # Example usage of the class and its methods
   a <- matrix(1:36, 6, 6)
   mmpr_matrix <- as.MMPR(a,nrow=6,ncol=6,precision="single")

   # Normal equal '=' will create a new pointer of the object, so any change in object A
   # will affect object B
   temp_mmpr_matrix = mmpr_matrix
   temp_mmpr_matrix[2,2] <- 500
   mmpr_matrix[2,2]           #500


   mmpr_matrix_copy <- MMPR.copy(mmpr_matrix)
   mmpr_matrix[2,2] <-100
   mmpr_matrix_copy[2,2] <- 200

   mmpr_matrix[2,2]           #100
   mmpr_matrix_copy[2,2]      #200
  }
}