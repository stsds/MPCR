## Up until R 2.15.0, the require("methods") is needed but (now)
## triggers an warning from R CMD check
#.onLoad <- function(libname, pkgname){
#    #require("methods")  ## needed with R <= 2.15.0
#    loadRcppModules()
#}


## For R 2.15.1 and later this also works. Note that calling loadModule() triggers
## a load action, so this does not have to be placed in .onLoad() or evalqOnLoad().
.onLoad <- function(libname, pkgname) {
  loadModule("MMPR", TRUE, loadNow = TRUE)
  loadModule("MMPRTile", TRUE, loadNow = TRUE)

  #--------------------------------------------------------------------------------
  # MPR Tile
  setMethod("[", signature(x = "Rcpp_MMPRTile"), function(x, i, j, drop = TRUE) {
    if (missing(j)) {
      stop("Please Provide a 2D Index")
    }else {
      i = i - 1
      j = j - 1
      ret <- x$MMPRTile.GetVal(i, j)
      ret
    }
  })

  setReplaceMethod("[", signature(x = "Rcpp_MMPRTile", value = "ANY"), function(x, i, j, ..., value) {
    if (missing(j)) {
      stop("Please Provide a 2D Index")
    }else {
      i = i - 1
      j = j - 1
      x$MMPRTile.SetVal(i, j, value)
    }
    x
  })

  #-------------------------- MPRTile Print ---------------------------------------
  setMethod("print", c(x = "Rcpp_MMPRTile"), function(x, ...) {
    x$MMPRTile.print()
  })
  #-------------------------- MPRTile Linear Algebra ------------------------------
  setMethod("chol", c(x = "Rcpp_MMPRTile"), MMPRTile.chol)

  #--------------------------------------------------------------------------------

  #------------------------------ MPR Class----------------------------------------
  setMethod("[", signature(x = "Rcpp_MMPR"), function(x, i, j, drop = TRUE) {
    if (missing(j)) {
      i = i - 1
      ret <- x$MMPR.GetVal(i)
      ret
    }else {
      i = i - 1
      j = j - 1
      ret <- x$MMPR.GetValMatrix(i, j)
      ret
    }
  })

  setReplaceMethod("[", signature(x = "Rcpp_MMPR", value = "ANY"), function(x, i, j, ..., value) {
    if (missing(j)) {
      i = i - 1
      x$MMPR.SetVal(i, value)
    }else {
      i = i - 1
      j = j - 1
      x$MMPR.SetValMatrix(i, j, value)
    }
    x
  })

  setMethod("[[", signature(x = "Rcpp_MMPR"), function(x, i, drop = TRUE) {
    i = i - 1
    ret <- x$MMPR.GetVal(i)
    ret
  })

  setReplaceMethod("[[", signature(x = "Rcpp_MMPR", value = "ANY"), function(x, i, ..., value) {
    i = i - 1
    x$MMPR.SetVal(i, value)
    x
  })


  # Basic Utilities
  # -----------------------------------------------------------------------------
  # trig   - Done
  # -----------------------------------------------------------------------------
  setMethod("sin", c(x = "Rcpp_MMPR"), MMPR.sin)
  setMethod("cos", c(x = "Rcpp_MMPR"), MMPR.cos)
  setMethod("tan", c(x = "Rcpp_MMPR"), MMPR.tan)
  setMethod("asin", c(x = "Rcpp_MMPR"), MMPR.asin)
  setMethod("acos", c(x = "Rcpp_MMPR"), MMPR.acos)
  setMethod("atan", c(x = "Rcpp_MMPR"), MMPR.atan)


  # -----------------------------------------------------------------------------
  # hyperbolic - Done
  # -----------------------------------------------------------------------------
  setMethod("sinh", c(x = "Rcpp_MMPR"), MMPR.sinh)
  setMethod("cosh", c(x = "Rcpp_MMPR"), MMPR.cosh)
  setMethod("tanh", c(x = "Rcpp_MMPR"), MMPR.tanh)
  setMethod("asinh", c(x = "Rcpp_MMPR"), MMPR.asinh)
  setMethod("acosh", c(x = "Rcpp_MMPR"), MMPR.acosh)
  setMethod("atanh", c(x = "Rcpp_MMPR"), MMPR.atanh)


  # -----------------------------------------------------------------------------
  # logs - Done
  # -----------------------------------------------------------------------------
  setMethod("exp", c(x = "Rcpp_MMPR"), MMPR.exp)
  setMethod("expm1", c(x = "Rcpp_MMPR"), MMPR.expm1)
  setMethod("log", c(x = "Rcpp_MMPR"), MMPR.log)
  setMethod("log2", c(x = "Rcpp_MMPR"), MMPR.log2)
  setMethod("log10", c(x = "Rcpp_MMPR"), MMPR.log10)


  # -----------------------------------------------------------------------------
  # misc - Done
  # -----------------------------------------------------------------------------
  setMethod("abs", c(x = "Rcpp_MMPR"), MMPR.abs)
  setMethod("sqrt", c(x = "Rcpp_MMPR"), MMPR.sqrt)


  # -----------------------------------------------------------------------------
  # special - Done
  # -----------------------------------------------------------------------------
  setMethod("gamma", c(x = "Rcpp_MMPR"), MMPR.gamma)
  setMethod("lgamma", c(x = "Rcpp_MMPR"), MMPR.lgamma)


  # -----------------------------------------------------------------------------
  # mathis - Done
  # -----------------------------------------------------------------------------
  setMethod("is.finite", c(x = "Rcpp_MMPR"), MMPR.is.finite)
  setMethod("is.infinite", c(x = "Rcpp_MMPR"), MMPR.is.infinite)
  setMethod("is.nan", c(x = "Rcpp_MMPR"), MMPR.is.nan)


  # -----------------------------------------------------------------------------
  # rounding - Done
  # -----------------------------------------------------------------------------
  setMethod("ceiling", c(x = "Rcpp_MMPR"), MMPR.ceiling)
  setMethod("floor", c(x = "Rcpp_MMPR"), MMPR.floor)
  setMethod("trunc", c(x = "Rcpp_MMPR"), MMPR.trunc)
  setMethod("round", c(x = "Rcpp_MMPR"), MMPR.round)


  # -----------------------------------------------------------------------------
  # Meta-Data - Done
  # -----------------------------------------------------------------------------
  setMethod("storage.mode", c(x = "Rcpp_MMPR"), MMPR.storage.mode)
  setMethod("typeof", c(x = "Rcpp_MMPR"), MMPR.typeof)





  # -----------------------------------------------------------------------------
  # Min-Max - Done
  # -----------------------------------------------------------------------------
  setMethod("min", c(x = "Rcpp_MMPR"), MMPR.min)
  setMethod("max", c(x = "Rcpp_MMPR"), MMPR.max)
  setMethod("which.min", c(x = "Rcpp_MMPR"), MMPR.which.min)
  setMethod("which.max", c(x = "Rcpp_MMPR"), MMPR.which.max)


  # -----------------------------------------------------------------------------
  # Dims - Done
  # -----------------------------------------------------------------------------
  setMethod("nrow", c(x = "Rcpp_MMPR"), MMPR.nrow)
  setMethod("ncol", c(x = "Rcpp_MMPR"), MMPR.ncol)

  # -----------------------------------------------------------------------------
  # Prints - Done
  # -----------------------------------------------------------------------------
  setMethod("print", c(x = "Rcpp_MMPR"), MMPR.print)
  setMethod("show", c(object = "Rcpp_MMPR"), MMPR.show)


  # -----------------------------------------------------------------------------
  # Basic Utilities - Done
  # -----------------------------------------------------------------------------
  setMethod("diag", c(x = "Rcpp_MMPR"), MMPR.diag)
  setMethod("rep", signature(x = "Rcpp_MMPR"), MMPR.rep)
  setMethod("sweep", c(x = "Rcpp_MMPR"), MMPR.sweep)
  setMethod("scale", c(x = "Rcpp_MMPR"), MMPR.scale)


  # Operators
  # -----------------------------------------------------------------------------
  # arithmetic - Done
  # -----------------------------------------------------------------------------
  setMethod("+", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformPlus(e2)
    ret
  })

  setMethod("-", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformMinus(e2)
    ret
  })

  setMethod("*", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformMult(e2)
    ret
  })

  setMethod("/", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformDiv(e2)
    ret
  })

  setMethod("^", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformPow(e2)
    ret
  })
  NULL
  # -----------------------------------------------------------------------------
  # Comparisons - Done
  # -----------------------------------------------------------------------------
  setMethod(">", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$GreaterThan(e2)
    ret
  })

  setMethod(">=", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$GreaterEqual(e2)
    ret
  })

  setMethod("<", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$LessThan(e2)
    ret
  })

  setMethod("<=", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$LessEqual(e2)
    ret
  })

  setMethod("==", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$EqualEqual(e2)
    ret
  })

  setMethod("!=", c(e1 = "Rcpp_MMPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$NotEqual(e2)
    ret
  })


  # Linear Algebra - Done
  # ---------------------


  setMethod("t", signature(x = "Rcpp_MMPR"), function(x) {
    ret <- MMPR.t(x)
    ret
  })

  setMethod("isSymmetric", signature(object = "Rcpp_MMPR"), function(object, ...) {
    ret <- MMPR.isSymmetric(object)
    ret
  })

  setMethod("chol", signature(x = "Rcpp_MMPR"), function(x, upper_triangle, ...) {
    if (missing(upper_triangle)) {
      upper_triangle = TRUE
    }
    ret <- MMPR.chol(x, upper_triangle)
    ret
  })

  setMethod("chol2inv", c(x = "Rcpp_MMPR"), function(x, size) {
    if (missing(size)) {
      size <- x$Col
    }
    ret <- MMPR.chol2inv(x, size)
    ret
  })


  setMethod("qr", c(x = "Rcpp_MMPR"), function(x,tol) {
    if(missing(tol)){
      tol= 1e-07
    }
    ret <- MMPR.qr(x,tol)
    names(ret) <- c("qr", "qraux", "pivot", "rank")
    ret
  })

  setMethod("qr.R", c(qr = "ANY"), function(qr, complete = FALSE) {

    if (class(qr) == "list") {
      if (length(qr) == 4 && class(qr[[2]]) == "Rcpp_MMPR") {
        if (missing(complete)) {
          complete = FALSE
        }
        ret <- MMPR.qr.R(qr$qr, complete)
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

    if (class(qr) == "list") {
      if (length(qr) == 4 && class(qr[[2]]) == "Rcpp_MMPR") {
        if (missing(Dvec)) {
          Dvec = NULL
        }
        if (missing(complete)) {
          complete = FALSE
        }
        ret <- MMPR.qr.Q(qr$qr, qr$qraux, complete, Dvec)
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
    if (class(qr) == "list") {
      if (length(qr) == 4 && class(qr[[2]]) == "Rcpp_MMPR") {
        ret <- MMPR.qr.qy(qr$qr, qr$qraux,y)
        ret

      }else {
        ret <- base::qr.qy(qr, y)
        ret
      }
    }

  })

  setMethod("qr.qty", c(qr = "ANY"), function(qr, y) {
    if (class(qr) == "list") {
      if (length(qr) == 4 && class(qr[[2]]) == "Rcpp_MMPR") {
        ret <- MMPR.qr.qty(qr$qr, qr$qraux,y)
        ret

      }else {
        ret <- base::qr.qty(qr, y)
        ret
      }
    }

  })

  setMethod("svd", c(x = "Rcpp_MMPR"), function(x, nu, nv) {
    if (missing(nu)) {
      nu = -1
    }
    if (missing(nv)) {
      nv = -1
    }
    ret <- MMPR.svd(x, nu, nv)
    names(ret) <- c("d", "u", "v")
    ret
  })

  setMethod("La.svd", c(x = "Rcpp_MMPR"), function(x, nu, nv) {
    if (missing(nu)) {
      nu = -1
    }
    if (missing(nv)) {
      nv = -1
    }
    ret <- MMPR.La.svd(x)
    names(ret) <- c("d", "u", "vt")
    ret
  })

  setMethod("crossprod", signature(x = "Rcpp_MMPR"), function(x, y = NULL) {
    ret <- MMPR.crossprod(x, y)
    ret
  })
  setMethod("tcrossprod", signature(x = "Rcpp_MMPR"), function(x, y = NULL) {
    ret <- MMPR.tcrossprod(x, y)
    ret
  })

  setMethod("%*%", signature(x = "Rcpp_MMPR", y = "Rcpp_MMPR"), MMPR.crossprod)


  setMethod("eigen", c(x = "Rcpp_MMPR"), function(x, only.values) {
    if (missing(only.values)) {
      only.values = FALSE
    }
    ret <- MMPR.eigen(x, only.values)
    if (length(ret) > 1) {
      names(ret) <- c("values", "vector")
    }else {
      names(ret) <- c("values")
    }
    ret
  })

  setMethod("solve", signature(a = "Rcpp_MMPR"), function(a, b, ...) {
    if (missing(b)) {
      b = NULL
    }
    ret <- MMPR.solve(a, b)
    ret
  })


  setMethod("backsolve", c(r = "Rcpp_MMPR", x = "Rcpp_MMPR"), function(r, x, k, upper.tri = FALSE, transpose = FALSE) {
    if (missing(k)) {
      k = -1
    }
    if (missing(upper.tri)) {
      upper.tri = TRUE
    }
    if (missing(transpose)) {
      transpose = FALSE
    }
    ret <- MMPR.backsolve(r, x, k, upper.tri, transpose)
    ret

  })
  setMethod("forwardsolve", c(l = "Rcpp_MMPR", x = "Rcpp_MMPR"),
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
              ret <- MMPR.forwardsolve(l, x, k, upper.tri, transpose)
              ret

            })


  setMethod("norm", c(x = "Rcpp_MMPR"), function(x, type) {
    if (missing(type)) {
      type = "O"
    }
    ret <- MMPR.norm(x, type)
    ret
  })

  setMethod("rcond", signature(x = "Rcpp_MMPR"), function(x, norm = "O", triangular = FALSE, ...) {
    if (missing(norm)) {
      norm = "O"
    }
    if (missing(triangular)) {
      triangular = FALSE
    }
    MMPR.rcond(x, norm, triangular)
  })
}