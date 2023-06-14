\name{scale}
\alias{scale}
\alias{scale,Rcpp_MMPR-method}
\title{scale}
\usage{
\S4method{scale}{Rcpp_MMPR}(x, center, scale)
}
\arguments{
\item{x}{A MMPR vector/matrix.}

\item{center, scale}{logical or MMPR objects}
}
\value{
A MMPR matrix.
}
\description{
Center/scale a MMPR vector/matrix.
}
\examples{
  \dontrun{
  library(MMPR)
    x <-as.MMPR(1:50,"single")
    x$ToMatrix(5, 10)
    temp_center_scale <- new(1:10,precision="double")
    z <- scale(x=temp_scale, center=FALSE, scale=temp_center_scale)
}
}
