\name{MMPR}
\alias{MMPR}
\title{MMPR S4 Class}

\description{
MMPR is a Multi-Precision Vector/Matrix.
You can create a vector/matrix with 3 different precisions (16-bit, 32-bit, 64-bit)
}

\section{Constructor}{
  \code{\link{new}} creates a new instance of zero values of the \code{MMPR} class.
    \cr
  \code{new(MMPR,size, "precision")}
  \describe{
  \item{\code{size}}{total number of values to allocate}
  \item{\code{precision}}{string to indicate the precision of MMPR object ("half","single","double")}
}
}



\section{Accessors}{
  The following accessors can be used to get the values of the slots:

  \describe{
    \item{\code{IsMatrix}}{Boolean to indicate whether the MMPR object is a vector or matrix}
        \item{\code{Size}}{total number of elements inside the object, (row*col) in the case of matrix , and number of elements in the case of vector}
        \item{\code{Row}}{Number of rows}
        \item{\code{Col}}{Number of cols}
  }
}

\section{Methods}{
  The following methods are available for objects of class \code{MMPR}:

  \subsection{PrintValues}{
    \code{PrintValues()}: Prints all the values stored in the matrix or vector ,plus meta-data about the object.
  }

  \subsection{ToMatrix}{
    \code{ToMatrix(row,col)}: Change the object representation to match the new Dimensions, no memory overhead.
  }

  \subsection{ToVector}{
      \code{ToVector()}: Changes the MMPR Matrix to vector , no memory overhead.
  }
}

\examples{
  \dontrun{
    # Example usage of the class and its methods
    mmpr_object <- new(MMPR,50,"single")

    mmpr_object$ToMatrix(5,10)
    mmpr_object$Row       #5
    mmpr_object$Col       #10
    mmpr_object$Size      #50
    mmpr_object$IsMatrix  #TRUE

    mmpr_object$PrintValues()
    mmpr_object$ToVector()

    mmpr_object

  }
}

\keyword{S4 class}
