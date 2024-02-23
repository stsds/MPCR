\name{23-Hyperbolic}
\alias{hyperbolic}
\alias{sinh,Rcpp_MPCR-method}
\alias{cosh,Rcpp_MPCR-method}
\alias{tanh,Rcpp_MPCR-method}
\alias{asinh,Rcpp_MPCR-method}
\alias{acosh,Rcpp_MPCR-method}
\alias{atanh,Rcpp_MPCR-method}

\alias{MPCR.sinh}
\alias{MPCR.cosh}
\alias{MPCR.tanh}
\alias{MPCR.asinh}
\alias{MPCR.acosh}
\alias{MPCR.atanh}

\title{Hyperbolic functions}
\description{
These functions give the obvious hyperbolic functions. They respectively compute the hyperbolic cosine, sine, tangent, and their inverses, arc-cosine, arc-sine, arc-tangent (or 'area cosine', etc).
}
\usage{
\S4method{sinh}{Rcpp_MPCR}(x)
\S4method{cosh}{Rcpp_MPCR}(x)
\S4method{tanh}{Rcpp_MPCR}(x)
\S4method{asinh}{Rcpp_MPCR}(x)
\S4method{acosh}{Rcpp_MPCR}(x)
\S4method{atanh}{Rcpp_MPCR}(x)
}
\arguments{
    \item{x}{
    An MPCR object.
    }
}
\value{
An MPCR object of the same dimensions as the input.
}
\examples{

    library(MPCR)

    mpcr_matrix <- as.MPCR(1:20,nrow=2,ncol=10,precision="single")
    x <- sinh(mpcr_matrix)

}