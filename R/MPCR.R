##########################################################################
# Copyright (c) 2023, King Abdullah University of Science and Technology
# All rights reserved.
# MPCR is an R package provided by the STSDS group at KAUST
##########################################################################


.onLoad <- function(libname, pkgname) {

  loadModule("MPCR", TRUE, loadNow = TRUE)

  utils::globalVariables(c("n", "p"))
  suppressMessages({
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


  # Linear Algebra - Done
  # ---------------------


  setMethod("t", signature(x = "Rcpp_MPCR"), function(x) {
    ret <- MPCR.t(x)
    ret
  })

  setMethod("isSymmetric", signature(object = "Rcpp_MPCR"), function(object, ...) {
    ret <- MPCR.isSymmetric(object)
    ret
  })

  setMethod("chol", signature(x = "Rcpp_MPCR"), function(x, upper_triangle, ...) {
    if (missing(upper_triangle)) {
      upper_triangle = TRUE
    }
    ret <- MPCR.chol(x, upper_triangle)
    ret
  })

  setMethod("chol2inv", c(x = "Rcpp_MPCR"), function(x, size) {
    if (missing(size)) {
      size <- x$Col
    }
    ret <- MPCR.chol2inv(x, size)
    ret
  })


  setMethod("qr", c(x = "Rcpp_MPCR"), function(x) {
    ret <- MPCR.qr(x)
    names(ret) <- c("qr", "qraux", "pivot", "rank")
    ret
  })

  setMethod("qr.R", c(qr = "ANY"), function(qr, complete = FALSE) {

    if (is(qr, "list")) {
      if (length(qr) == 4 && is(qr[[2]], "Rcpp_MPCR")) {
        if (missing(complete)) {
          complete = FALSE
        }
        ret <- MPCR.qr.R(qr$qr, complete)
        ret
      }else {
        ret <- base::qr.R(qr, complete)
        ret
      }

    }else {
      ret <- base::qr.R(qr, complete)
      ret
    }
  })

  setMethod("qr.Q", c(qr = "ANY"), function(qr, complete = FALSE, Dvec) {

    if (is(qr, "list")) {
      if (length(qr) == 4 && is(qr[[2]], "Rcpp_MPCR")) {
        if (missing(Dvec)) {
          Dvec = NULL
        }
        if (missing(complete)) {
          complete = FALSE
        }
        ret <- MPCR.qr.Q(qr$qr, qr$qraux, complete, Dvec)
        ret

      }else {
        ret <- base::qr.Q(qr, complete, Dvec)
        ret
      }
    }
    else {
      ret <- base::qr.Q(qr, complete, Dvec)
      ret
    }
  })

  setMethod("qr.qy", c(qr = "ANY"), function(qr, y) {
    if (is(qr, "list")) {
      if (length(qr) == 4 && is(qr[[2]], "Rcpp_MPCR")) {
        ret <- MPCR.qr.qy(qr$qr, qr$qraux, y)
        ret

      }else {
        ret <- base::qr.qy(qr, y)
        ret
      }
    }

  })

  setMethod("qr.qty", c(qr = "ANY"), function(qr, y) {
    if (is(qr, "list")) {
      if (length(qr) == 4 && is(qr[[2]], "Rcpp_MPCR")) {
        ret <- MPCR.qr.qty(qr$qr, qr$qraux, y)
        ret

      }else {
        ret <- base::qr.qty(qr, y)
        ret
      }
    }

  })

  setMethod("svd", c(x = "Rcpp_MPCR"), function(x, nu, nv) {
    if (missing(nu)) {
      nu = -1
    }
    if (missing(nv)) {
      nv = -1
    }
    ret <- MPCR.svd(x, nu, nv)
    names(ret) <- c("d", "u", "v")
    ret
  })

  setMethod("La.svd", c(x = "Rcpp_MPCR"), function(x, nu = min(n, p), nv = min(n, p)) {
    n = x$Row
    p = x$Col

    if (missing(nu)) {
      nu = -1
    }
    if (missing(nv)) {
      nv = -1
    }
    ret <- MPCR.La.svd(x, nu, nv)
    names(ret) <- c("d", "u", "vt")
    ret
  })

  setMethod("crossprod", signature(x = "Rcpp_MPCR"), function(x, y = NULL) {
    ret <- MPCR.crossprod(x, y)
    ret
  })
  setMethod("tcrossprod", signature(x = "Rcpp_MPCR"), function(x, y = NULL) {
    ret <- MPCR.tcrossprod(x, y)
    ret
  })

  setMethod("%*%", signature(x = "Rcpp_MPCR", y = "Rcpp_MPCR"), MPCR.crossprod)


  setMethod("eigen", c(x = "Rcpp_MPCR"), function(x, only.values) {
    if (missing(only.values)) {
      only.values = FALSE
    }
    ret <- MPCR.eigen(x, only.values)
    if (length(ret) > 1) {
      names(ret) <- c("values", "vector")
    }else {
      names(ret) <- c("values")
    }
    ret
  })

  setMethod("solve", signature(a = "Rcpp_MPCR"), function(a, b,internal_precision="same", ...) {
    if (missing(b)) {
      b = NULL
    }
    ret <- MPCR.solve(a, b,internal_precision)
    ret
  })


  setMethod("backsolve", c(r = "Rcpp_MPCR", x = "Rcpp_MPCR"), function(r, x, k, upper.tri = FALSE, transpose = FALSE) {
    if (missing(k)) {
      k = -1
    }
    if (missing(upper.tri)) {
      upper.tri = TRUE
    }
    if (missing(transpose)) {
      transpose = FALSE
    }
    ret <- MPCR.backsolve(r, x, k, upper.tri, transpose)
    ret

  })
  setMethod("forwardsolve", c(l = "Rcpp_MPCR", x = "Rcpp_MPCR"),
            function(l, x, k, upper.tri = FALSE, transpose = FALSE) {
              if (missing(k)) {
                k = -1
              }
              if (missing(upper.tri)) {
                upper.tri = FALSE
              }
              if (missing(transpose)) {
                transpose = FALSE
              }
              ret <- MPCR.forwardsolve(l, x, k, upper.tri, transpose)
              ret

            })


  setMethod("norm", c(x = "Rcpp_MPCR"), function(x, type) {
    if (missing(type)) {
      type = "O"
    }
    ret <- MPCR.norm(x, type)
    ret
  })

  setMethod("rcond", signature(x = "Rcpp_MPCR"), function(x, norm = "O", triangular = FALSE, ...) {
    if (missing(norm)) {
      norm = "O"
    }
    if (missing(triangular)) {
      triangular = FALSE
    }
    MPCR.rcond(x, norm, triangular)
  })
  })
}