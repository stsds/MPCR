\name{DeepCopy}
\alias{DeepCopy}
\title{DeepCopy}


\section{MMPR DeepCopy}{
    \description{
    Create a copy from MMPR object.
    }

    \subsection{copy}{
      \code{MMPR.copy(x)}: Create a new copy of MMPR Matrix/Vector
      \describe{
      \item{\code{x}}{MMPR Matrix/Vector }
    }}

}

\section{MMPRTile DeepCopy}{
    \description{
     Create a copy from MMPRTile object.
    }

   \subsection{copy}{
     \code{MMPRTile.copy(x)}: Create a new copy of MMPRTile Matrix
      \describe{
         \item{\code{x}}{MMPRTile Matrix }
      }
   }
}


\examples{
  \dontrun{
   # Example usage of the class and its methods
   a <- matrix(1:36, 6, 6)
   mmpr_matrix <- as.MMPR(a,nrow=6,ncol=6,precision="single")

   # Normal equal will create a new pointer of the object, so any change in object A
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