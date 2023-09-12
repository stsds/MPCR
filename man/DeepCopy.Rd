\name{08-Copy}
\alias{copy}
\title{copy}

\description{
 Functions for copying MPCR objects.
}

\section{MPCR deep copy}{
    \describe{
   Create a copy of an MPCR object. Typically, using 'equal' creates a new pointer for the object, resulting in any modifications made to object one affecting object two as well.
    }

    \subsection{copy}{
    \cr
      \code{MPCR.copy(x)}: Create a new copy of an MPCR object.
      \describe{
      \item{\code{x}}{MPCR object. }
    }}

}

\section{MPCRTile deep copy}{
    \describe{
    Create a duplicate of an MPCRTile object. Usually, using 'equal' creates a new pointer for the object, causing any modifications made to object one to affect object two as well.
    }

   \subsection{copy}{
   \cr
     \code{MPCRTile.copy(x)}: Create a new copy of an MPCRTile matrix.
      \describe{
         \item{\code{x}}{MPCRTile matrix. }
      }
   }
}


\examples{
  \dontrun{
   # Example usage of the class and its methods
   a <- matrix(1:36, 6, 6)
   MPCR_matrix <- as.MPCR(a,nrow=6,ncol=6,precision="single")

   # Normal equal '=' will create a new pointer of the object, so any change in object A
   # will affect object B
   temp_MPCR_matrix = MPCR_matrix
   temp_MPCR_matrix[2,2] <- 500
   MPCR_matrix[2,2]           #500


   MPCR_matrix_copy <- MPCR.copy(MPCR_matrix)
   MPCR_matrix[2,2] <-100
   MPCR_matrix_copy[2,2] <- 200

   MPCR_matrix[2,2]           #100
   MPCR_matrix_copy[2,2]      #200
  }
}