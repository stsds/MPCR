## Up until R 2.15.0, the require("methods") is needed but (now)
## triggers an warning from R CMD check
#.onLoad <- function(libname, pkgname){
#    #require("methods")  ## needed with R <= 2.15.0
#    loadRcppModules()
#}


## For R 2.15.1 and later this also works. Note that calling loadModule() triggers
## a load action, so this does not have to be placed in .onLoad() or evalqOnLoad().

.onLoad <- function(libname, pkgname) {
  loadModule("MPR", TRUE, loadNow = TRUE)

  RPlus <- setMethod("+", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformPlus(e2)
    ret
  })

  RMinus <- setMethod("-", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformMinus(e2)
    ret
  })

  RMult <- setMethod("*", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformMult(e2)
    ret
  })

  RDiv <- setMethod("/", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformDiv(e2)
    ret
  })

  RPow <- setMethod("^", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformPow(e2)
    ret
  })

  RGreater <- setMethod(">", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$GreaterThan(e2)
    ret
  })

  RGreaterEqual <- setMethod(">=", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$GreaterEqual(e2)
    ret
  })

  RLess <- setMethod("<", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$LessThan(e2)
    ret
  })

  RLessEqual <- setMethod("<=", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$LessEqual(e2)
    ret
  })

  REqualEqual <- setMethod("==", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$EqualEqual(e2)
    ret
  })

  RNotEqual <- setMethod("!=", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$NotEqual(e2)
    ret
  })
}