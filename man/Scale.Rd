\name{19-Scale}
\alias{scale}
\alias{scale,Rcpp_MPCR-method}
\title{scale}
\usage{
\S4method{scale}{Rcpp_MPCR}(x, center, scale)
}
\arguments{
\item{x}{An MPCR object.}

\item{center, scale}{Logical or MPCR objects.}
}
\value{
An MPCR matrix.
}
\description{
Center or scale an MPCR object.
}
\examples{
  \donttest{
  library(MPCR)
    input <-as.MPCR(1:50,precision="single")
    x$ToMatrix(5, 10)
    temp_center_scale <- new(1:10,precision="double")
    z <- scale(x=input, center=FALSE, scale=temp_center_scale)
}
}
