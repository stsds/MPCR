library(MPCR)

# Matern covariance function
cov.matern <- function(x, nu, a, sigma_sq){
  if(nu == 0.5) return(sigma_sq*exp(-x/a))
  
  # Store matrix dimensions if input is matrix
  is_matrix <- is.matrix(x)
  if(is_matrix) {
    nr <- nrow(x)
    nc <- ncol(x)
  }
  
  x <- c(x/a)
  output <- rep(1, length(x))
  idx <- x > 0
  if(sum(idx) > 0) {
    x1 <- x[idx]
    output[idx] <- (1/((2^(nu-1))*gamma(nu))) * (x1^nu) * besselK(x1, nu)
  }
  
  # Return as matrix if input was matrix
  if(is_matrix) {
    output <- matrix(output, nr, nc)
  }
  
  sigma_sq * output
}

# Setup spatial field
M <- 100; n <- M^2
locs <- expand.grid(x=(0:(M-1))/(M-1), y=(0:(M-1))/(M-1))
D <- as.matrix(dist(locs))

# True parameters: nu=1, a=0.05, sigma^2=1
theta_true <- c(1, 0.05, 1)
cov_true <- cov.matern(D, theta_true[1], theta_true[2], theta_true[3])

# Simulate data
set.seed(4)
L <- t(chol(cov_true))
z <- L %*% rnorm(n)

# Negative log-likelihood function
nll <- function(pars, prec='R-Double') {
  # Transform parameters to ensure valid ranges
  nu <- 2 * plogis(pars[1])  # nu in (0,2)
  a <- exp(pars[2])           # a > 0
  sigma2 <- exp(pars[3])      # sigma^2 > 0
  
  # Compute covariance
  V <- cov.matern(D, nu, a, sigma2)
  
  # Simple trick: add nugget for single precision stability
  if(grepl('Single', prec)) {
    diag(V) <- diag(V) + 1e-6
  }
  
  use_mpcr <- grepl('MPCR', prec)
  if(use_mpcr) {

	  # Precision
	  p <- if (prec == "MPCR-Single-CPU" || prec == "MPCR-Single-GPU") "single" else "double"

	  # Normalize placement to expected values
	  op_place  <- if (prec == "MPCR-Double-CPU" || prec == "MPCR-Double-CPU") "CPU" else "GPU"

	  # Set MPCR operation placement once
	  MPCR.SetOperationPlacement(placement = op_place)
	  print(op_place)	
	  # Convert once (pass placement only when GPU is requested)
	  if (op_place == "GPU") {
		  V_mpcr <- as.MPCR(V, n, n, p, placement= "GPU")
		  z_mpcr <- as.MPCR(z, n, 1, p, placement= "GPU")
	  } else {
		  V_mpcr <- as.MPCR(V, n, n, p)
		  z_mpcr <- as.MPCR(z, n, 1, p)
	  }

          MPCR.SetOperationPlacement(placement = op_place)

	  # Cholesky decomposition (upper triangular in MPCR)
	  U <- chol(V_mpcr)
	  L <- t(U)  # Convert to lower triangular

	  # Log determinant
	  log_det <- log(diag(U))$Sum() * 2

	  # Quadratic form using forward solve: L*w = z, then quad = w'*w
	  w <- forwardsolve(L, z_mpcr)
	  quad <- w$SquareSum()

  } else {
	  # R computation
	  L <- t(chol(V))  # Lower triangular
	  log_det <- 2*sum(log(diag(L)))

	  # Quadratic form using forward solve
	  w <- forwardsolve(L, z)
	  quad <- sum(w^2)
  }

  # Negative log-likelihood
  0.5*quad + 0.5*log_det + 0.5*n*log(2*pi)
}

# Optimization
init <- c(-0.3, -1.5, -0.3)  # Initial values

cat("\nStarting MLE optimization for n =", n, "\n")
cat("=========================================\n\n")

# R-Double optimization
cat("R-Double optimization...\n")
t0 <- Sys.time()
fit_R <- optim(init, nll, prec='R-Double', control=list(maxit=1000, trace=3))
time_R <- difftime(Sys.time(), t0, units="secs")

# Extract R estimates
theta_R <- c(2*plogis(fit_R$par[1]), exp(fit_R$par[2]), exp(fit_R$par[3]))
cat("\nR-Double results:\n")
cat("  NLL:", fit_R$value, "\n")
cat("  Estimates:", sprintf("(%.7f, %.7f, %.7f)", theta_R[1], theta_R[2], theta_R[3]), "\n")
cat("  Time:", round(time_R, 2), "secs\n\n")

# MPCR-Double-CPU optimization
cat("MPCR-Double-CPU optimization...\n")
t0 <- Sys.time()
fit_MPCR_D <- optim(init, nll, prec='MPCR-Double-CPU', control=list(maxit=1000, trace=3))
time_MPCR_D <- difftime(Sys.time(), t0, units="secs")

# Extract MPCR-Double-CPU estimates
theta_MPCR_D <- c(2*plogis(fit_MPCR_D$par[1]), exp(fit_MPCR_D$par[2]), exp(fit_MPCR_D$par[3]))
cat("\nMPCR-Double-CPU results:\n")
cat("  NLL:", fit_MPCR_D$value, "\n")
cat("  Estimates:", sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_D[1], theta_MPCR_D[2], theta_MPCR_D[3]), "\n")
cat("  Time:", round(time_MPCR_D, 2), "secs\n\n")


########

# MPCR-Double-CPU optimization
cat("MPCR-Double-GPU optimization...\n")
t0 <- Sys.time()
fit_MPCR_DG <- optim(init, nll, prec='MPCR-Double-GPU', control=list(maxit=1000, trace=3))
time_MPCR_DG <- difftime(Sys.time(), t0, units="secs")

# Extract MPCR-Double-CPU estimates
theta_MPCR_DG <- c(2*plogis(fit_MPCR_DG$par[1]), exp(fit_MPCR_DG$par[2]), exp(fit_MPCR_DG$par[3]))
cat("\nMPCR-Double-GPU results:\n")
cat("  NLL:", fit_MPCR_DG$value, "\n")
cat("  Estimates:", sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_DG[1], theta_MPCR_DG[2], theta_MPCR_DG[3]), "\n")
cat("  Time:", round(time_MPCR_DG, 2), "secs\n\n")
###

########

# MPCR-Double-CPU optimization
cat("MPCR-Single-GPU optimization...\n")
t0 <- Sys.time()
fit_MPCR_SG <- optim(init, nll, prec='MPCR-Single-GPU', control=list(maxit=1000, trace=3))
time_MPCR_SG <- difftime(Sys.time(), t0, units="secs")

# Extract MPCR-Double-CPU estimates
theta_MPCR_SG <- c(2*plogis(fit_MPCR_SG$par[1]), exp(fit_MPCR_SG$par[2]), exp(fit_MPCR_SG$par[3]))
cat("\nMPCR-Single-GPU results:\n")
cat("  NLL:", fit_MPCR_SG$value, "\n")
cat("  Estimates:", sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_SG[1], theta_MPCR_SG[2], theta_MPCR_SG[3]), "\n")
cat("  Time:", round(time_MPCR_SG, 2), "secs\n\n")
###



# MPCR-Single-CPU optimization
cat("MPCR-Single-CPU optimization (with nugget for stability)...\n")
t0 <- Sys.time()
fit_MPCR_S <- optim(init, nll, prec='MPCR-Single-CPU', control=list(maxit=1000, trace=3))
time_MPCR_S <- difftime(Sys.time(), t0, units="secs")

# Extract MPCR-Single-CPU estimates
theta_MPCR_S <- c(2*plogis(fit_MPCR_S$par[1]), exp(fit_MPCR_S$par[2]), exp(fit_MPCR_S$par[3]))
cat("\nMPCR-Single-CPU results:\n")
cat("  NLL:", fit_MPCR_S$value, "\n")
cat("  Estimates:", sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_S[1], theta_MPCR_S[2], theta_MPCR_S[3]), "\n")
cat("  Time:", round(time_MPCR_S, 2), "secs\n\n")

# Summary table
cat("\n=========================================\n")
cat("Summary of MLE results for n =", n, "\n")
cat("=========================================\n\n")
cat(sprintf("%-15s %10s %45s %15s\n", "Precision", "NLL", "Parameter Estimates", "Time"))
cat(sprintf("%-15s %10.0f %45s %15s\n", 
	    "R-Double", fit_R$value, 
	    sprintf("(%.7f, %.7f, %.7f)", theta_R[1], theta_R[2], theta_R[3]),
	    paste0(round(time_R, 2), " secs")))
cat(sprintf("%-15s %10.0f %45s %15s\n", 
	    "MPCR-Double-CPU", fit_MPCR_D$value,
	    sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_D[1], theta_MPCR_D[2], theta_MPCR_D[3]),
	    paste0(round(time_MPCR_D, 2), " secs")))
cat(sprintf("%-15s %10.0f %45s %15s\n", 
	    "MPCR-Single-CPU", fit_MPCR_S$value,
	    sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_S[1], theta_MPCR_S[2], theta_MPCR_S[3]),
	    paste0(round(time_MPCR_S, 2), " secs")))

cat(sprintf("%-15s %10.0f %45s %15s\n",
            "MPCR-Double-GPU", fit_MPCR_D$value,
            sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_DG[1], theta_MPCR_DG[2], theta_MPCR_DG[3]),
            paste0(round(time_MPCR_DG, 2), " secs")))
cat(sprintf("%-15s %10.0f %45s %15s\n",
            "MPCR-Single-GPU", fit_MPCR_S$value,
            sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_SG[1], theta_MPCR_SG[2], theta_MPCR_SG[3]),
            paste0(round(time_MPCR_SG, 2), " secs")))



cat("\nTrue parameters:", sprintf("(%.7f, %.7f, %.7f)", theta_true[1], theta_true[2], theta_true[3]), "\n")
cat("\nSpeedup MPCR-Double-CPU vs R-Double:", sprintf("%.1fx\n", as.numeric(time_R/time_MPCR_D)))
cat("Speedup MPCR-Single-CPU vs R-Double:", sprintf("%.1fx\n", as.numeric(time_R/time_MPCR_S)))

# Save results
save(fit_R, fit_MPCR_D, fit_MPCR_S, theta_R, theta_MPCR_D, theta_MPCR_S, 
     time_R, time_MPCR_D, time_MPCR_S, theta_true,
     file="section6.2.RData")
cat("\nResults saved to section6.2.RData\n")
