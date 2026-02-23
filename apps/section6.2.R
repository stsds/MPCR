library(MPCR)

# Matern covariance function
cov.matern <- function(x, nu, a, sigma_sq) {
	nu =0.5
	if (nu == 0.5) return(sigma_sq * exp(-x/a))

	# scale once
	xa <- x / a

	# constants once
	c0 <- 1 / (2^(nu - 1) * gamma(nu))

	out <- numeric(length(xa))
	out[] <- 1

	idx <- xa > 0
	if (any(idx)) {
		x1 <- xa[idx]
		out[idx] <- c0 * (x1^nu) * besselK(x1, nu)
	}

	dim(out) <- dim(x)  # preserves matrix shape if x is matrix
	sigma_sq * out
}


# Setup spatial field
M <- 150; n <- M^2
locs <- expand.grid(x=(0:(M-1))/(M-1), y=(0:(M-1))/(M-1))
D <- as.matrix(dist(locs))

# True parameters: nu=1, a=0.05, sigma^2=1
theta_true <- c(1, 0.1)
cov_true <- cov.matern(D, 0.5, theta_true[2], theta_true[1])

# Simulate data
set.seed(4)
L <- t(chol(cov_true))
z <- L %*% rnorm(n)


# Generate PDF plot of simulated field using quilt.plot
#library(fields)

#pdf("MPCR_JSS_application_mle_simulated_values.pdf", width=7, height=7)

#quilt.plot(locs$x, locs$y, z,
#           xlab = expression(s[x]),
#           ylab = expression(s[y]),
#           zlim = c(-3, 3))

#dev.off()

#cat("\nPlot saved to MPCR_JSS_application_mle_simulated_values.pdf\n")



# Negative log-likelihood function
nll <- function(pars, prec='R-Double') {
	# Transform parameters to ensure valid ranges
	nu <- 0.5 #2 * plogis(pars[1])  # nu in (0,2)
	a <- exp(pars[2])           # a > 0
	sigma2 <- exp(pars[1])      # sigma^2 > 0

	# Compute covariance
	t_chol <- system.time({
		inv_a <- 1 / a
V <- sigma2 * exp(-D * inv_a)

	})
	cat("[MPCR]", prec, "generate the matrix:", t_chol["elapsed"], "sec\n")

	# Simple trick: add nugget for single precision stability
	if(grepl('Single', prec)) {
		diag(V) <- diag(V) + 1e-6
	}

	use_mpcr <- grepl('MPCR', prec)
	if(use_mpcr) {

		# Precision
		p <- if (prec == "MPCR-Single-CPU" || prec == "MPCR-Single-GPU") "single" else "double"

		# Normalize placement to expected values
		op_place  <- if (prec == "MPCR-Single-CPU" || prec == "MPCR-Double-CPU") "CPU" else "GPU"

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


		t_chol <- system.time({
			U <- chol(V_mpcr)
			L <- t(U)
		})
		cat("[MPCR]", prec, "chol:", t_chol["elapsed"], "sec\n")

		diag <- system.time({
			# Log determinant
			log_det <- log(diag(U))$Sum() * 2
		})
		cat("[MPCR]", prec, "diag:", diag["elapsed"], "sec\n")

		fs <- system.time({
			# Quadratic form using forward solve: L*w = z, then quad = w'*w
			w <- forwardsolve(L, z_mpcr)
			quad <- w$SquareSum()
		})
		cat("[MPCR]", prec, "fs:", fs["elapsed"], "sec\n")


	} else {
		# R computation
		t_chol <- system.time({
			U <- chol(V)   # Step 1: Cholesky (upper triangular)
			L <- t(U)      # Step 2: transpose to lower triangular
		})

		cat("[MPCR]", prec, "chol:", t_chol["elapsed"], "sec\n")
		log_det <- 2*sum(log(diag(L)))

		# Quadratic form using forward solve
		w <- forwardsolve(L, z)
		quad <- sum(w^2)
	}

	# Negative log-likelihood
	0.5*quad + 0.5*log_det + 0.5*n*log(2*pi)
}

# Optimization
init <- c(1.5, -0.3)  # Initial values

cat("\nStarting MLE optimization for n =", n, "\n")
cat("=========================================\n\n")


# R-Double optimization
cat("R-Double optimization...\n")
t0 <- Sys.time()
fit_R <- optim(init, nll, prec='R-Double', control=list(maxit=1000, trace=2))
time_R <- difftime(Sys.time(), t0, units="secs")

# Extract R estimates
theta_R <- c(2*plogis(fit_R$par[1]), exp(fit_R$par[2]), 0.5)
cat("\nR-Double results:\n")
cat("  NLL:", fit_R$value, "\n")
cat("  Estimates:", sprintf("(%.7f, %.7f, %.7f)", theta_R[1], theta_R[2], 0.5), "\n")
cat("  Time:", round(time_R, 2), "secs\n\n")



# MPCR-Double-CPU optimization
cat("MPCR-Double-CPU optimization...\n")
t0 <- Sys.time()
fit_MPCR_D <- optim(init, nll, prec='MPCR-Double-CPU', control=list(maxit=1000, trace=2))
time_MPCR_D <- difftime(Sys.time(), t0, units="secs")

# Extract MPCR-Double-CPU estimates
theta_MPCR_D <- c(2*plogis(fit_MPCR_D$par[1]), exp(fit_MPCR_D$par[2]), 0.5)
cat("\nMPCR-Double-CPU results:\n")
cat("  NLL:", fit_MPCR_D$value, "\n")
cat("  Estimates:", sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_D[1], theta_MPCR_D[2], 0.5), "\n")
cat("  Time:", round(time_MPCR_D, 2), "secs\n\n")


########

# MPCR-Double-GPU optimization
cat("MPCR-Double-GPU optimization...\n")
t0 <- Sys.time()
fit_MPCR_DG <- optim(init, nll, prec='MPCR-Double-GPU', control=list(maxit=1000, trace=2))
time_MPCR_DG <- difftime(Sys.time(), t0, units="secs")

# Extract MPCR-Double-CPU estimates
theta_MPCR_DG <- c(2*plogis(fit_MPCR_DG$par[1]), exp(fit_MPCR_DG$par[2]), 0.5)
cat("\nMPCR-Double-GPU results:\n")
cat("  NLL:", fit_MPCR_DG$value, "\n")
cat("  Estimates:", sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_DG[1], theta_MPCR_DG[2], 0.5), "\n")
cat("  Time:", round(time_MPCR_DG, 2), "secs\n\n")
###

########

# MPCR-Double-CPU optimization
cat("MPCR-Single-GPU optimization...\n")
t0 <- Sys.time()
fit_MPCR_SG <- optim(init, nll, prec='MPCR-Single-GPU', control=list(maxit=1000, trace=2))
time_MPCR_SG <- difftime(Sys.time(), t0, units="secs")

# Extract MPCR-Double-CPU estimates
theta_MPCR_SG <- c(2*plogis(fit_MPCR_SG$par[1]), exp(fit_MPCR_SG$par[2]), 0.5)
cat("\nMPCR-Single-GPU results:\n")
cat("  NLL:", fit_MPCR_SG$value, "\n")
cat("  Estimates:", sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_SG[1], theta_MPCR_SG[2], 0.5), "\n")
cat("  Time:", round(time_MPCR_SG, 2), "secs\n\n")
###



# MPCR-Single-CPU optimization
cat("MPCR-Single-CPU optimization (with nugget for stability)...\n")
t0 <- Sys.time()
fit_MPCR_S <- optim(init, nll, prec='MPCR-Single-CPU', control=list(maxit=1000, trace=2))
time_MPCR_S <- difftime(Sys.time(), t0, units="secs")

# Extract MPCR-Single-CPU estimates
theta_MPCR_S <- c(2*plogis(fit_MPCR_S$par[1]), exp(fit_MPCR_S$par[2]), 0.5)
cat("\nMPCR-Single-CPU results:\n")
cat("  NLL:", fit_MPCR_S$value, "\n")
cat("  Estimates:", sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_S[1], theta_MPCR_S[2], 0.5), "\n")
cat("  Time:", round(time_MPCR_S, 2), "secs\n\n")

# Summary table
cat("\n=========================================\n")
cat("Summary of MLE results for n =", n, "\n")
cat("=========================================\n\n")
cat(sprintf("%-15s %10s %45s %15s\n", "Precision", "NLL", "Parameter Estimates", "Time"))
cat(sprintf("%-15s %10.0f %45s %15s\n", 
	    "R-Double", fit_R$value, 
	    sprintf("(%.7f, %.7f, %.7f)", theta_R[1], theta_R[2], 0.5),
	    paste0(round(time_R, 2), " secs")))
cat(sprintf("%-15s %10.0f %45s %15s\n", 
	    "MPCR-Double-CPU", fit_MPCR_D$value,
	    sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_D[1], theta_MPCR_D[2], 0.5),
	    paste0(round(time_MPCR_D, 2), " secs")))
cat(sprintf("%-15s %10.0f %45s %15s\n", 
	    "MPCR-Single-CPU", fit_MPCR_S$value,
	    sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_S[1], theta_MPCR_S[2], 0.5),
	    paste0(round(time_MPCR_S, 2), " secs")))

cat(sprintf("%-15s %10.0f %45s %15s\n",
	    "MPCR-Double-GPU", fit_MPCR_D$value,
	    sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_DG[1], theta_MPCR_DG[2], 0.5),
	    paste0(round(time_MPCR_DG, 2), " secs")))
cat(sprintf("%-15s %10.0f %45s %15s\n",
	    "MPCR-Single-GPU", fit_MPCR_S$value,
	    sprintf("(%.7f, %.7f, %.7f)", theta_MPCR_SG[1], theta_MPCR_SG[2], 0.5),
	    paste0(round(time_MPCR_SG, 2), " secs")))



cat("\nTrue parameters:", sprintf("(%.7f, %.7f, %.7f)", theta_true[1], theta_true[2], 0.5), "\n")
#cat("\nSpeedup MPCR-Double-CPU vs R-Double:", sprintf("%.1fx\n", as.numeric(time_R/time_MPCR_D)))
#cat("Speedup MPCR-Single-CPU vs R-Double:", sprintf("%.1fx\n", as.numeric(time_R/time_MPCR_S)))

# Save results
save(fit_R, fit_MPCR_D, fit_MPCR_S, theta_R, theta_MPCR_D, theta_MPCR_S, 
     time_R, time_MPCR_D, time_MPCR_S, theta_true,
     file="section6.2.RData")
cat("\nResults saved to section6.2.RData\n")



# Generate PDF plot of simulated field using quilt.plot
library(fields)

pdf("MPCR_JSS_application_mle_simulated_values.pdf", width=7, height=7)

quilt.plot(locs$x, locs$y, z,
	   xlab = expression(s[x]),
	   ylab = expression(s[y]),
	   zlim = c(-3, 3))

dev.off()

cat("\nPlot saved to MPCR_JSS_application_mle_simulated_values.pdf\n")

