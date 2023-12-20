##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################


.onLoad <- function(libname, pkgname) {

  loadModule("MPCR", TRUE, loadNow = TRUE)
  loadModule("MPCRTile", TRUE, loadNow = TRUE)

  suppressMessages({
    #--------------------------------------------------------------------------------
    # MPR Tile
    setMethod("[", signature(x = "Rcpp_MPCRTile"), function(x, i, j, drop = TRUE) {
      if (missing(j)) {
        stop("Please Provide a 2D Index")
      }else {
        i = i - 1
        j = j - 1
        ret <- x$MPCRTile.GetVal(i, j)
        ret
      }
    })

    setReplaceMethod("[", signature(x = "Rcpp_MPCRTile", value = "ANY"), function(x, i, j, ..., value) {
      if (missing(j)) {
        stop("Please Provide a 2D Index")
      }else {
        i = i - 1
        j = j - 1
        x$MPCRTile.SetVal(i, j, value)
      }
      x
    })

    #-------------------------- MPCRTile Print ---------------------------------------
    setMethod("print", c(x = "Rcpp_MPCRTile"), function(x, ...) {
      x$MPCRTile.print()
    })
    #--------------------------------------------------------------------------------

    #------------------------------ MPR Class----------------------------------------
    setMethod("[", signature(x = "Rcpp_MPCR"), function(x, i, j, drop = TRUE) {
      if (missing(j)) {
        i = i - 1
        ret <- x$MPCR.GetVal(i)
        ret
      }else {
        i = i - 1
        j = j - 1
        ret <- x$MPCR.GetValMatrix(i, j)
        ret
      }
    })

    setReplaceMethod("[", signature(x = "Rcpp_MPCR", value = "ANY"), function(x, i, j, ..., value) {
      if (missing(j)) {
        i = i - 1
        x$MPCR.SetVal(i, value)
      }else {
        i = i - 1
        j = j - 1
        x$MPCR.SetValMatrix(i, j, value)
      }
      x
    })

    setMethod("[[", signature(x = "Rcpp_MPCR"), function(x, i, drop = TRUE) {
      i = i - 1
      ret <- x$MPCR.GetVal(i)
      ret
    })

    setReplaceMethod("[[", signature(x = "Rcpp_MPCR", value = "ANY"), function(x, i, ..., value) {
      i = i - 1
      x$MPCR.SetVal(i, value)
      x
    })


    # Basic Utilities
    # -----------------------------------------------------------------------------
    # trig   - Done
    # -----------------------------------------------------------------------------
    setMethod("sin", c(x = "Rcpp_MPCR"), MPCR.sin)
    setMethod("cos", c(x = "Rcpp_MPCR"), MPCR.cos)
    setMethod("tan", c(x = "Rcpp_MPCR"), MPCR.tan)
    setMethod("asin", c(x = "Rcpp_MPCR"), MPCR.asin)
    setMethod("acos", c(x = "Rcpp_MPCR"), MPCR.acos)
    setMethod("atan", c(x = "Rcpp_MPCR"), MPCR.atan)


    # -----------------------------------------------------------------------------
    # hyperbolic - Done
    # -----------------------------------------------------------------------------
    setMethod("sinh", c(x = "Rcpp_MPCR"), MPCR.sinh)
    setMethod("cosh", c(x = "Rcpp_MPCR"), MPCR.cosh)
    setMethod("tanh", c(x = "Rcpp_MPCR"), MPCR.tanh)
    setMethod("asinh", c(x = "Rcpp_MPCR"), MPCR.asinh)
    setMethod("acosh", c(x = "Rcpp_MPCR"), MPCR.acosh)
    setMethod("atanh", c(x = "Rcpp_MPCR"), MPCR.atanh)


    # -----------------------------------------------------------------------------
    # logs - Done
    # -----------------------------------------------------------------------------
    setMethod("exp", c(x = "Rcpp_MPCR"), MPCR.exp)
    setMethod("expm1", c(x = "Rcpp_MPCR"), MPCR.expm1)
    setMethod("log", c(x = "Rcpp_MPCR"), MPCR.log)
    setMethod("log2", c(x = "Rcpp_MPCR"), MPCR.log2)
    setMethod("log10", c(x = "Rcpp_MPCR"), MPCR.log10)


    # -----------------------------------------------------------------------------
    # misc - Done
    # -----------------------------------------------------------------------------
    setMethod("abs", c(x = "Rcpp_MPCR"), MPCR.abs)
    setMethod("sqrt", c(x = "Rcpp_MPCR"), MPCR.sqrt)


    # -----------------------------------------------------------------------------
    # special - Done
    # -----------------------------------------------------------------------------
    setMethod("gamma", c(x = "Rcpp_MPCR"), MPCR.gamma)
    setMethod("lgamma", c(x = "Rcpp_MPCR"), MPCR.lgamma)


    # -----------------------------------------------------------------------------
    # mathis - Done
    # -----------------------------------------------------------------------------
    setMethod("is.finite", c(x = "Rcpp_MPCR"), MPCR.is.finite)
    setMethod("is.infinite", c(x = "Rcpp_MPCR"), MPCR.is.infinite)
    setMethod("is.nan", c(x = "Rcpp_MPCR"), MPCR.is.nan)


    # -----------------------------------------------------------------------------
    # rounding - Done
    # -----------------------------------------------------------------------------
    setMethod("ceiling", c(x = "Rcpp_MPCR"), MPCR.ceiling)
    setMethod("floor", c(x = "Rcpp_MPCR"), MPCR.floor)
    setMethod("trunc", c(x = "Rcpp_MPCR"), MPCR.trunc)
    setMethod("round", c(x = "Rcpp_MPCR"), MPCR.round)


    # -----------------------------------------------------------------------------
    # Meta-Data - Done
    # -----------------------------------------------------------------------------
    setMethod("storage.mode", c(x = "Rcpp_MPCR"), MPCR.storage.mode)
    setMethod("typeof", c(x = "Rcpp_MPCR"), MPCR.typeof)


    # -----------------------------------------------------------------------------
    # Min-Max - Done
    # -----------------------------------------------------------------------------
    setMethod("min", c(x = "Rcpp_MPCR"), MPCR.min)
    setMethod("max", c(x = "Rcpp_MPCR"), MPCR.max)
    setMethod("which.min", c(x = "Rcpp_MPCR"), MPCR.which.min)
    setMethod("which.max", c(x = "Rcpp_MPCR"), MPCR.which.max)


    # -----------------------------------------------------------------------------
    # Dims - Done
    # -----------------------------------------------------------------------------
    setMethod("nrow", c(x = "Rcpp_MPCR"), MPCR.nrow)
    setMethod("ncol", c(x = "Rcpp_MPCR"), MPCR.ncol)

    # -----------------------------------------------------------------------------
    # Prints - Done
    # -----------------------------------------------------------------------------
    setMethod("print", c(x = "Rcpp_MPCR"), MPCR.print)
    setMethod("show", c(object = "Rcpp_MPCR"), MPCR.show)


    # -----------------------------------------------------------------------------
    # Basic Utilities - Done
    # -----------------------------------------------------------------------------
    setMethod("diag", c(x = "Rcpp_MPCR"), MPCR.diag)
    setMethod("rep", signature(x = "Rcpp_MPCR"), MPCR.rep)
    setMethod("sweep", c(x = "Rcpp_MPCR"), MPCR.sweep)
    setMethod("scale", c(x = "Rcpp_MPCR"), MPCR.scale)


    # Operators
    # -----------------------------------------------------------------------------
    # arithmetic - Done
    # -----------------------------------------------------------------------------
    setMethod("+", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$PerformPlus(e2)
      ret
    })

    setMethod("-", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$PerformMinus(e2)
      ret
    })

    setMethod("*", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$PerformMult(e2)
      ret
    })

    setMethod("/", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$PerformDiv(e2)
      ret
    })

    setMethod("^", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$PerformPow(e2)
      ret
    })
    NULL
    # -----------------------------------------------------------------------------
    # Comparisons - Done
    # -----------------------------------------------------------------------------
    setMethod(">", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$GreaterThan(e2)
      ret
    })

    setMethod(">=", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$GreaterEqual(e2)
      ret
    })

    setMethod("<", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$LessThan(e2)
      ret
    })

    setMethod("<=", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$LessEqual(e2)
      ret
    })

    setMethod("==", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$EqualEqual(e2)
      ret
    })

    setMethod("!=", c(e1 = "Rcpp_MPCR", e2 = "ANY"), function(e1, e2) {
      ret <- e1$NotEqual(e2)
      ret
    })

  })
}