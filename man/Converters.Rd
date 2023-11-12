\name{02-Converters}
\alias{Converters}
\title{Converters}

\description{
 Converters from R to MPCR objects and vice-versa.
}
\section{MPCR Converter}{
    \describe{
    Convert R object to MPCR object.
    }

   \subsection{MPCR converters}{
   \cr
     \code{as.MPCR(data,nrow = 0,ncol = 0,precision)}: Converts R object to MPCR object.
      \cr
      \describe{
         \item{\code{data}}{R matrix/vector.}
         \item{\code{nrow}}{Number of rows of the new MPCR matrix, \bold{default = zero} which means a vector will be created.}
         \item{\code{ncol}}{Number of cols of the new MPCR matrix, \bold{default = zero} which means a vector will be created.}
         \item{\code{precision}}{String indicates the precision of the new MPCR object (half, single, or double).}
      }
   }
}

\section{R Converter}{
    \describe{
    Convert an MPCR object to R object.
    }

   \subsection{R vector converter}{
   \cr
     \code{MPCR.ToNumericVector(x)}: Converts an MPCR object to a numeric R vector.
      \describe{
         \item{\code{x}}{MPCR object.}
      }
   }

      \subsection{R matrix converter}{
      \cr
        \code{MPCR.ToNumericMatrix(x)}: Converts an MPCR object to a numeric R matrix.
         \describe{
            \item{\code{x}}{MPCR object.}
         }
      }
}

\value{
An MPCR or R numeric vector/matrix.
}


\examples{
   # Example usage of the class and its methods
   library(MPCR)
   a <- matrix(1:36, 6, 6)
   MPCR_matrix <- as.MPCR(a,nrow=6,ncol=6,precision="single")
   r_vector <- MPCR.ToNumericVector(MPCR_matrix)
   r_vector
   r_matrix <- MPCR.ToNumericMatrix(MPCR_matrix)
   r_matrix
}