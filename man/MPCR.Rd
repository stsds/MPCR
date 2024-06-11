\name{01-MPCR}
\alias{MPCR}
\alias{Rcpp_MPCR-class}
\alias{MPCR.Serialize}
\alias{MPCR.DeSerialize}
\title{MPCR S4 Class}

\description{
MPCR is a multi-precision vector/matrix, that enables the creation of vector/matrix with three different precisions (16-bit (half), 32-bit(single), and 64-bit(double)) on CPU and GPU.
MPCR uses smart cache memory allocation system that minimize the number of memory transfers required during the
transition from CPU to GPU operations and vice verse. So the object at some point can be allocated on both CPU and GPU,
 in case, the user needed to free specific allocation, helper functions can be used.
}

\section{Constructor}{
  \code{\link{new}} Creates a new instance of zero values of the \code{MPCR} class.
  \code{new(MPCR,size, "precision","placement")}
  \describe{
  \item{\code{size}}{The total number of values for which memory needs to be allocated.}
  \item{\code{precision}}{String to indicate the precision of MPCR object ("half","single", or "double").}
  \item{\code{placement}}{String to indicate whether the allocation should be made on CPU (default) or GPU ("CPU","GPU") .}
}
}



\section{Accessors}{
  The following accessors can be used to get the values of the slots:

  \describe{
    \item{\code{IsMatrix}}{Boolean to indicate whether the MPCR object is a vector or matrix.}
        \item{\code{Size}}{Total number of elements inside the object, (row*col) in the case of matrix, and number of elements in the case of vector.}
        \item{\code{Row}}{Number of rows.}
        \item{\code{Col}}{Number of cols.}
  }
}

\section{Methods}{
  The following methods are available for objects of class \code{MPCR}:

  \subsection{PrintValues}{
    \code{PrintValues()}: Prints all the values stored in the matrix or vector, along with metadata about the object.
  }

  \subsection{ToMatrix}{
    \code{ToMatrix(row,col)}: Changes the object representation to match the new dimensions, no memory overhead.
  }

  \subsection{ToVector}{
      \code{ToVector()}: Changes the MPCR matrix to vector, no memory overhead.
  }

  \subsection{IsGPUAllocated}{
        \code{IsGPUAllocated()}: Returns TRUE if the MPCR object is allocated on GPU.
  }

  \subsection{IsCPUAllocated}{
        \code{IsCPUAllocated()}: Returns TRUE if the MPCR object is allocated on CPU.
  }

  \subsection{FreeGPU}{
        \code{FreeGPU()}: Free the data allocated on GPU.
  }

  \subsection{FreeCPU}{
        \code{FreeCPU()}: Free the data allocated on CPU.
  }
}

\value{
MPCR object (constructor - accessors - methods)
}

\examples{
  \donttest{
    # Example usage of the class and its methods
    library(MPCR)
    MPCR_object <- new(MPCR,50,"single","CPU")

    MPCR_object$ToMatrix(5,10)
    MPCR_object$Row       #5
    MPCR_object$Col       #10
    MPCR_object$Size      #50
    MPCR_object$IsMatrix  #TRUE

    MPCR_object$PrintValues()
    MPCR_object$ToVector()
    MPCR_object$IsCPUAllocated() #TRUE
    MPCR_object$IsGPUAllocated() #FALSE

    MPCR_object
  }
}

\keyword{S4 class}
