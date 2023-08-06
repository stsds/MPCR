\name{Diagonal}
\alias{diag}
\alias{diag,Rcpp_MMPR-method}
\title{diag}
\usage{
\S4method{diag}{Rcpp_MMPR}(x)
}
\arguments{
\item{x}{An MMPR matrix.}
}
\value{
An MMPR vector contains the main diagonal of the matrix.
}
\description{
Returns the diagonal of an MMPR matrix.
}
\examples{
  \dontrun{
    library(MMPR)
    x <- as.MMPR(1:16,4,4,"single")
    diag_vals <- diag(x)
    }

}
