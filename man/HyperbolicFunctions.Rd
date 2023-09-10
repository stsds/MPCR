\name{23-Hyperbolic}
\alias{hyperbolic}
\alias{sinh,Rcpp_MMPR-method}
\alias{cosh,Rcpp_MMPR-method}
\alias{tanh,Rcpp_MMPR-method}
\alias{asinh,Rcpp_MMPR-method}
\alias{acosh,Rcpp_MMPR-method}
\alias{atanh,Rcpp_MMPR-method}
\title{Hyperbolic functions}
\description{
These functions give the obvious hyperbolic functions. They respectively compute the hyperbolic cosine, sine, tangent, and their inverses, arc-cosine, arc-sine, arc-tangent (or 'area cosine', etc).
}
\usage{
\S4method{sinh}{Rcpp_MMPR}(x)
\S4method{cosh}{Rcpp_MMPR}(x)
\S4method{tanh}{Rcpp_MMPR}(x)
\S4method{asinh}{Rcpp_MMPR}(x)
\S4method{acosh}{Rcpp_MMPR}(x)
\S4method{atanh}{Rcpp_MMPR}(x)
}
\arguments{
    \item{x}{
    An MMPR object.
    }
}
\value{
An MMPR object of the same dimensions as the input.
}
\examples{
    \dontrun{
    library(MMPR)

    mppr_matrix <- as.MMPR(1:20,nrow=2,ncol=10,precision="single")
    x <- sinh(mppr_matrix)
    }
}