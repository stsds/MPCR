\name{40-Context Handling}
\alias{Context Handling}
\alias{MPCR.SetOperationPlacement,Rcpp_MPCR-method}
\alias{MPCR.GetOperationPlacement,Rcpp_MPCR-method}

\alias{MPCR.SetOperationPlacement}
\alias{MPCR.GetOperationPlacement}

\title{Context Handling}

\description{
The Context Placement handling mechanism used by MPCR manages the dispatch of operations to either the CPU or GPU. When the operation placement is set to "GPU," all operations with GPU implementations (e.g., Linear Algebra functions) will execute on the GPU. If an operation is not supported on the GPU, it will execute on the CPU without altering the Context Placement for subsequent code execution.

By default, the placement is set to the CPU at the start of execution. To switch to GPU placement, users must use specific functions to set and get the current placement.

The Context Placement mechanism does not control initial memory allocation. For optimal performance, users should be aware of where data is allocated.
 If the user is uncertain about data allocation, the package will automatically manage allocation and data movement between the CPU and GPU.
  It employs an internal caching mechanism to minimize memory transfers, although data might be allocated on both the CPU and GPU during the object's lifetime. Helper functions are available to check if memory is allocated on the CPU/GPU and to free memory on either one.
}

\section{Set Operation Context}{
  \code{\link{MPCR.SetOperationPlacement}} Set the placement for the up-coming flow of code ( "CPU", "GPU").
  \code{MPCR.SetOperationPlacement(placement)}
  \describe{
  \item{\code{placement}}{String to indicate on which hardware should the up-coming flow of code be executed.}
}
}

\section{Get Operation Context}{
  \code{\link{MPCR.GetOperationPlacement}} Get the placement used currently for dispatching operations ( "CPU", "GPU").
  \code{MPCR.GetOperationPlacement()}
  \describe{
  returns the placement currently used for operations.
}
}

\value{
 Operation Context (Setting and Getting).
}

\examples{
  library(MPCR)
  values <- c(3.12393, -1.16854, -0.304408, -2.15901,
              -1.16854, 1.86968, 1.04094, 1.35925, -0.304408,
              1.04094, 4.43374, 1.21072, -2.15901, 1.35925, 1.21072, 5.57265)


  x <- new(MPCR, 16, "float","GPU") # Data will be allocated on GPU
  y <- new(MPCR, 16, "float","GPU")

  # Since this two for loops changes the data inside the MPCR objects,
  # the GPU memory will be freed.

  for (val in 1:16) {
    x[[val]] <- values[[val]]
    y[[val]] <- values[[val]]

  }


  # At this point only CPU memory is allocated.

  x$ToMatrix(4, 4)
  y$ToMatrix(4, 4)
  paste("X and Y values")
  x$PrintValues()  # CPU Function
  y$PrintValues()  # CPU Function

  MPCR.SetOperationPlacement("GPU") # Set Function Dispatching to GPU
  cat("----------------------- CrossProduct C=XY --------------------\n")

  # GPU Cuda Kernel, The data will be automatically copied to GPU.
  crossproduct <- crossprod(x, y)

   # Data won't be moved to CPU, since crossprod didn't change the content.
  x$PrintValues()  # CPU Function
  y$PrintValues()  # CPU Function

  # CPU Function, so the data will be copied to CPU after finalizing the GPU Call.
  crossproduct$PrintValues()




}