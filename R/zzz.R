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
  loadModule("MPRTile", TRUE, loadNow = TRUE)

  #--------------------------------------------------------------------------------
  # MPR Tile
  setMethod("[", signature(x = "Rcpp_MPRTile"), function(x, i, j, drop = TRUE) {
    if (missing(j)) {
      stop("Please Provide a 2D Index")
    }else {
      i = i - 1
      j = j - 1
      ret <- x$MPRTile.GetVal(i, j)
      ret
    }
  })

  setReplaceMethod("[", signature(x = "Rcpp_MPRTile", value = "ANY"), function(x, i, j, ..., value) {
    if (missing(j)) {
      stop("Please Provide a 2D Index")
    }else {
      i = i - 1
      j = j - 1
      x$MPRTile.SetVal(i, j, value)
    }
    x
  })

  #-------------------------- MPRTile Print ---------------------------------------
  setMethod("print", c(x = "Rcpp_MPRTile"), function(x, ...) {
    x$MPRTile.print()
  })
  #-------------------------- MPRTile Linear Algebra ------------------------------
  setMethod("chol", c(x = "Rcpp_MPRTile"), MPRTile.chol)



  #------------------------------ MPR Class--------------------------------------------
  setMethod("[", signature(x = "Rcpp_MPR"), function(x, i, j, drop = TRUE) {
    if (missing(j)) {
      i = i - 1
      ret <- x$MPR.GetVal(i)
      ret
    }else {
      i = i - 1
      j = j - 1
      ret <- x$MPR.GetValMatrix(i, j)
      ret
    }
  })

  setReplaceMethod("[", signature(x = "Rcpp_MPR", value = "ANY"), function(x, i, j, ..., value) {
    if (missing(j)) {
      i = i - 1
      x$MPR.SetVal(i, value)
    }else {
      i = i - 1
      j = j - 1
      x$MPR.SetValMatrix(i, j, value)
    }
    x
  })

  setMethod("[[", signature(x = "Rcpp_MPR"), function(x, i, drop = TRUE) {
    i = i - 1
    ret <- x$MPR.GetVal(i)
    ret
  })

  setReplaceMethod("[[", signature(x = "Rcpp_MPR", value = "ANY"), function(x, i, ..., value) {
    i = i - 1
    x$MPR.SetVal(i, value)
    x
  })


  # Basic Utilities
  # ----------------

  #-----------------------------------------------------------------------------------------------#
  setMethod("abs",c(x="Rcpp_MPR"),MPR.abs)
  setMethod("sqrt",c(x="Rcpp_MPR"),MPR.sqrt)
  setMethod("ceiling",c(x="Rcpp_MPR"),MPR.ceiling)
  setMethod("floor",c(x="Rcpp_MPR"),MPR.floor)
  setMethod("trunc",c(x="Rcpp_MPR"),MPR.trunc)
  setMethod("exp",c(x="Rcpp_MPR"),MPR.exp)
  setMethod("expm1",c(x="Rcpp_MPR"),MPR.expm1)
  setMethod("gamma",c(x="Rcpp_MPR"),MPR.gamma)
  setMethod("lgamma",c(x="Rcpp_MPR"),MPR.lgamma)
  setMethod("is.finite",c(x="Rcpp_MPR"),MPR.is.finite)
  setMethod("is.infinite",c(x="Rcpp_MPR"),MPR.is.infinite)
  setMethod("is.nan",c(x="Rcpp_MPR"),MPR.is.nan)
  setMethod("log10",c(x="Rcpp_MPR"),MPR.log10)
  setMethod("log2",c(x="Rcpp_MPR"),MPR.log2)
  setMethod("sin",c(x="Rcpp_MPR"),MPR.sin)
  setMethod("cos",c(x="Rcpp_MPR"),MPR.cos)
  setMethod("tan",c(x="Rcpp_MPR"),MPR.tan)
  setMethod("asin",c(x="Rcpp_MPR"),MPR.asin)
  setMethod("acos",c(x="Rcpp_MPR"),MPR.acos)
  setMethod("atan",c(x="Rcpp_MPR"),MPR.atan)
  setMethod("sinh",c(x="Rcpp_MPR"),MPR.sinh)
  setMethod("cosh",c(x="Rcpp_MPR"),MPR.cosh)
  setMethod("tanh",c(x="Rcpp_MPR"),MPR.tanh)
  setMethod("asinh",c(x="Rcpp_MPR"),MPR.asinh)
  setMethod("acosh",c(x="Rcpp_MPR"),MPR.acosh)
  setMethod("atanh",c(x="Rcpp_MPR"),MPR.atanh)
  setMethod("round",c(x="Rcpp_MPR"),MPR.round)
  setMethod("log",c(x="Rcpp_MPR"),MPR.log)

  setMethod("diag",c(x="Rcpp_MPR"),MPR.diag)
  setMethod("min",c(x="Rcpp_MPR"),MPR.min)
  setMethod("max",c(x="Rcpp_MPR"),MPR.max)
  setMethod("which.min",c(x="Rcpp_MPR"),MPR.which.min)
  setMethod("which.max",c(x="Rcpp_MPR"),MPR.which.max)
  setMethod("nrow",c(x="Rcpp_MPR"),MPR.nrow)
  setMethod("ncol",c(x="Rcpp_MPR"),MPR.ncol)
  setMethod("typeof",c(x="Rcpp_MPR"),MPR.typeof)
  setMethod("na.exclude",signature(object="Rcpp_MPR"),MPR.na.exclude)
  setMethod("na.omit",c(object="Rcpp_MPR"),MPR.na.omit)
  setMethod("object.size",c(x="Rcpp_MPR"),MPR.object.size)
  setMethod("show",c(object="Rcpp_MPR"),MPR.show)
  setMethod("storage.mode",c(x="Rcpp_MPR"),MPR.storage.mode)
  setMethod("rep",signature(x="Rcpp_MPR"),MPR.rep)
  setMethod("sweep",c(x="Rcpp_MPR"),MPR.sweep)
  setMethod("scale",c(x="Rcpp_MPR"),MPR.scale)
  setMethod("print", c(x = "Rcpp_MPR"), MPR.print)


  # Operators
  # ------------

  setMethod("+", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformPlus(e2)
    ret
  })

  setMethod("-", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformMinus(e2)
    ret
  })

  setMethod("*", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformMult(e2)
    ret
  })

  setMethod("/", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformDiv(e2)
    ret
  })

  setMethod("^", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$PerformPow(e2)
    ret
  })

  setMethod(">", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$GreaterThan(e2)
    ret
  })

  setMethod(">=", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$GreaterEqual(e2)
    ret
  })

  setMethod("<", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$LessThan(e2)
    ret
  })

  setMethod("<=", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$LessEqual(e2)
    ret
  })

  setMethod("==", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$EqualEqual(e2)
    ret
  })

  setMethod("!=", c(e1 = "Rcpp_MPR", e2 = "ANY"), function(e1, e2) {
    ret <- e1$NotEqual(e2)
    ret
  })

  # Linear Algebra
  # ---------------------

  setMethod("svd", c(x = "Rcpp_MPR"), function(x, nu, nv) {
    if (missing(nu)) {
      nu = -1
    }
    if (missing(nv)) {
      nv = -1
    }
    ret <- MPR.svd(x, nu, nv)
    names(ret) <- c("d", "u", "v")
    ret
  })

  setMethod("La.svd", c(x = "Rcpp_MPR"), function(x, nu, nv) {
    if (missing(nu)) {
      nu = -1
    }
    if (missing(nv)) {
      nv = -1
    }
    ret <- MPR.La.svd(x)
    names(ret) <- c("d", "u", "vt")
    ret
  })


  setMethod("qr", c(x = "Rcpp_MPR"), function(x) {
    ret <- MPR.qr(x)
    names(ret) <- c("qr", "qraux", "pivot", "rank")
    ret
  })

  setMethod("eigen", c(x = "Rcpp_MPR"), function(x, only.values) {
    if (missing(only.values)) {
      only.values = FALSE
    }
    ret <- MPR.eigen(x, only.values)
    if (length(ret) > 1) {
      names(ret) <- c("values", "vector")
    }else {
      names(ret) <- c("values")
    }
    ret
  })

  setMethod("backsolve", c(r = "Rcpp_MPR", x = "Rcpp_MPR"), function(r, x, k, upper.tri = FALSE, transpose = FALSE) {
    if (missing(k)) {
      k = -1
    }
    if (missing(upper.tri)) {
      upper.tri = TRUE
    }
    if (missing(transpose)) {
      transpose = FALSE
    }
    ret <- MPR.backsolve(r, x, k, upper.tri, transpose)
    ret

  })
  setMethod("forwardsolve", c(l = "Rcpp_MPR", x = "Rcpp_MPR"),
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
              ret <- MPR.forwardsolve(l, x, k, upper.tri, transpose)
              ret

            })
  setMethod("chol", signature(x = "Rcpp_MPR"), function(x,upper_triangle, ...) {
    if(missing(upper_triangle)){
      upper_triangle=TRUE
    }
    ret <- MPR.chol(x,upper_triangle)
    ret
  })
  setMethod("chol2inv", c(x = "Rcpp_MPR"), function(x, size) {
    if (missing(size)) {
      size <- x$Col
    }
    ret <- MPR.chol2inv(x, size)
    ret
  })
  setMethod("crossprod", signature(x = "Rcpp_MPR"), function(x, y = NULL) {
    ret <- MPR.crossprod(x, y)
    ret
  })
  setMethod("tcrossprod", signature(x = "Rcpp_MPR"), function(x, y = NULL) {
    ret <- MPR.tcrossprod(x, y)
    ret
  })

  setMethod("%*%", signature(x = "Rcpp_MPR", y = "Rcpp_MPR"), MPR.crossprod)

  setMethod("isSymmetric", signature(object = "Rcpp_MPR"), function(object, ...) {
    ret <- MPR.isSymmetric(object)
    ret
  })
  setMethod("norm", c(x = "Rcpp_MPR"), function(x, type) {
    if (missing(type)) {
      type = "O"
    }
    ret <- MPR.norm(x, type)
    ret
  })
  setMethod("solve", signature(a = "Rcpp_MPR"), function(a, b, ...) {
    if (missing(b)) {
      b = NULL
    }
    ret <- MPR.solve(a, b)
    ret
  })
  setMethod("t", signature(x = "Rcpp_MPR"), function(x) {
    ret <- MPR.t(x)
    ret
  })
  setMethod("rcond", signature(x = "Rcpp_MPR"), function(x, norm = "O", triangular = FALSE, ...) {
    if (missing(norm)) {
      norm = "O"
    }
    if (missing(triangular)) {
      triangular = FALSE
    }
    MPR.rcond(x, norm, triangular)
  })
  setMethod("qr.R", c(qr = "ANY"), function(qr, complete = FALSE) {

    if (class((qr) == "list")) {
      if (length(qr) == 4 && class(qr[[2]]) == "Rcpp_MPR") {
        if (missing(complete)) {
          complete = FALSE
        }
        ret <- MPR.qr.R(qr$qr, complete)
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
      if (length(qr) == 4 && class(qr[[2]]) == "Rcpp_MPR") {
        if (missing(Dvec)) {
          Dvec = NULL
        }
        if (missing(complete)) {
          complete = FALSE
        }
        ret <- MPR.qr.Q(qr$qr, qr$qraux, complete, Dvec)
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


}