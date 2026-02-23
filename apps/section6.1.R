library(MPCR)
parallel::detectCores(logical = FALSE)   # physical cores
parallel::detectCores(logical = TRUE)    # logical (hyperthreads)

# Function to run MALA with different precisions
run_mala <- function(M, prec='R-Double', I=2000) {
  n <- M^2 # Setup: spatial grid
  locs <- expand.grid(x=(0:(M-1))/(M-1), y=(0:(M-1))/(M-1))
  D <- as.matrix(dist(locs))
  
  # Target distribution: mean=0, covariance=exp(-D/0.5)
  mu <- rep(0, n); sig <- exp(-D/0.5)
  # Preconditioning matrix for MALA
  pre_M <- exp(-D/0.05); h <- 0.01

  # Convert to MPCR if needed (consistent precision throughout)
  use_mpcr <- grepl('MPCR', prec)
  if(use_mpcr) {
	  # Precision
	  p <- if (prec == "MPCR-Single-CPU" || prec == "MPCR-Single-GPU") "single" else "double"
	  # Hardware placement to expected values
	  op_place  <- if (prec == "MPCR-Single-CPU" || prec == "MPCR-Double-CPU") "CPU" else "GPU"

	  # Convert ALL matrices to MPCR format at once
	  mu <- as.MPCR(mu, n, 1, p, placement=op_place)
	  sig <- as.MPCR(sig, n, n, p, placement=op_place)
	  pre_M <- as.MPCR(pre_M, n, n, p, placement=op_place)
	  MPCR.SetOperationPlacement(placement = op_place)
	  # Precompute matrices and keep in MPCR format
	  sig.inv <- solve(sig)
	  pre_M_h <- pre_M$PerformMult(h)

	  # Compute Cholesky and keep in MPCR format
	  L <- t(chol(pre_M_h))

	  # Compute inverse and keep in MPCR format
	  pre_M_inv <- solve(pre_M_h)

  } else {
	  # BLAS backends
	  n_phys <- parallel::detectCores(logical = FALSE)
	  print(n_phys)
	  # Sanity check
	  cat("Physical cores:", n_phys, "\n")
	  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
		  cat("BLAS threads:", RhpcBLASctl::blas_get_num_procs(), "\n")
	  }
	  Sys.setenv(OPENBLAS_NUM_THREADS = n_phys)
	  Sys.setenv(MKL_NUM_THREADS      = n_phys)
	  Sys.setenv(VECLIB_MAXIMUM_THREADS = n_phys)  # macOS Accelerate
	  # Standard R operations
	  sig.inv <- solve(sig)
	  L <- t(chol(h*pre_M))
	  pre_M_inv <- solve(h*pre_M)
  }

  # Gradient and gradient step functions
  grad <- function(x) {
	  if(use_mpcr) {
		  sig.inv %*% x  # MPCR handles this natively
	  } else {
		  sig.inv %*% x
	  }
  }

  step <- function(x, t) {
	  if(use_mpcr) {
		  # Use MPCR's native operations
		  grad_x <- grad(x)
		  pre_M_grad <- pre_M %*% grad_x
		  x - pre_M_grad$PerformMult(t)
	  } else {
		  x - t * pre_M %*% grad(x)
	  }
  }

  # Helper function for quadratic forms - optimized for MPCR
  quadform <- function(x, A, y) {
	  if(use_mpcr) {
		  # Compute x'Ay more efficiently
		  Ay <- A %*% y
		  result <- t(x) %*% Ay
		  MPCR.ToNumericVector(result)[1]  # Convert only the scalar result
	  } else {
		  as.numeric(t(x) %*% A %*% y)
	  }
  }

  # Initialize chain
  set.seed(1234); z <- runif(n)
  if(use_mpcr) z <- as.MPCR(z, n, 1, p,  placement=op_place)  # Use consistent precision

  # Store trace only for M=120 (for visualization)
  if(M == 120) {
	  trace <- matrix(NA, n, I)
  }

  # Track acceptance rate for diagnostics
  accept_count <- 0

  # MALA iterations
  t0 <- Sys.time()
  for(i in 1:I) {
	  set.seed(i)

	  # Generate proposal
	  if(use_mpcr) {
		  eps <- as.MPCR(rnorm(n), n, 1, p, placement=op_place)
		  z_mean <- step(z, 0.5*h)
		  z_prop <- z_mean + (L %*% eps)
	  } else {
		  eps <- rnorm(n)
		  z_prop <- step(z, 0.5*h) + L %*% eps
	  }

	  # Compute log probability ratios using optimized quadform
	  if(use_mpcr) {
		  # Compute differences once
		  z_prop_mu <- z_prop - mu
		  z_mu <- z - mu

		  # Use quadform for efficient computation
		  p_prop <- -0.5 * quadform(z_prop_mu, sig.inv, z_prop_mu)
		  p_curr <- -0.5 * quadform(z_mu, sig.inv, z_mu)

		  # Compute reverse proposal probabilities
		  z_prop_mean <- step(z_prop, 0.5*h)
		  z_mean_curr <- step(z, 0.5*h)

		  diff_curr <- z - z_prop_mean
		  diff_prop <- z_prop - z_mean_curr

		  q_curr <- -0.5 * quadform(diff_curr, pre_M_inv, diff_curr)
		  q_prop <- -0.5 * quadform(diff_prop, pre_M_inv, diff_prop)

	  } else {
		  # Standard R operations
		  p_prop <- -0.5 * as.numeric(t(z_prop-mu) %*% sig.inv %*% (z_prop-mu))
		  p_curr <- -0.5 * as.numeric(t(z-mu) %*% sig.inv %*% (z-mu))
		  q_curr <- -0.5 * as.numeric(t(z-step(z_prop,0.5*h)) %*% pre_M_inv %*% 
					      (z-step(z_prop,0.5*h)))
		  q_prop <- -0.5 * as.numeric(t(z_prop-step(z,0.5*h)) %*% pre_M_inv %*% 
					      (z_prop-step(z,0.5*h)))
	  }

	  # Metropolis-Hastings accept/reject
	  log_ratio <- p_prop - p_curr + q_curr - q_prop
	  if(runif(1) < exp(min(0, log_ratio))) {
		  z <- z_prop
		  accept_count <- accept_count + 1
	  }

	  # Store trace only for M=120 - convert only once per iteration
	  if(M == 120) {
		  trace[,i] <- if(use_mpcr) MPCR.ToNumericVector(z) else z
	  }
  }

  elapsed_time <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  accept_rate <- accept_count / I

  result <- list(time=elapsed_time, accept_rate=accept_rate)
  #  if(M == 120) result$trace <- trace
  if(M == 120) result$trace <- trace
  result
}

# Define grid sizes to test
M_sizes <- c(30, 60, 90, 120)
#M_sizes <- c(5, 6, 7, 120)
#M_sizes <- c(120)
precs <- c('R-Double', 'MPCR-Double-CPU', 'MPCR-Single-CPU', 'MPCR-Double-GPU', 'MPCR-Single-GPU')
#precs <- c('R-Double', 'R-Double', 'R-Double', 'R-Double', 'R-Double')
# Initialize timing matrix
timing_results <- matrix(NA, length(M_sizes), length(precs))
res_M120 <- list() # Store M=120 results for visualization

cat("\nRunning MALA with different grid sizes and precisions...\n")
for(i in seq_along(M_sizes)) {
	M <- M_sizes[i]
	cat(sprintf("\nGrid size M=%d (n=%d):\n", M, M^2))

	for(j in seq_along(precs)) {
		cat(sprintf("  %s...", precs[j]))
		result <- run_mala(M, precs[j])
		timing_results[i,j] <- result$time

		# Store M=120 results for visualization
		if(M == 120) {
			res_M120[[precs[j]]] <- result
		}

		cat(sprintf(" %.2f secs (Accept: %.1f%%)\n", 
			    result$time, result$accept_rate*100))
	}
}

colnames(timing_results) <- precs
rownames(timing_results) <- paste0("M=", M_sizes)

cat("\nTiming Results (seconds):\n")
print(timing_results)

# Visualization for M=120 only
M <- 120; I <- 2000
pdf("mh_results_figure_4.pdf", width=10, height=6)
par(mfrow=c(2,3), mar=c(5,5,3,4)) 
cols <- rev(rainbow(100, start=0, end=4/6)) 

# Panel 1: Initial z0
set.seed(1234); z0 <- matrix(runif(M^2), M, M)
image(z0, main=expression(bold(z[0])), col=cols, cex.lab=1.5,
      xlab=expression(s[x]), ylab=expression(s[y]), axes=FALSE)
axis(1); axis(2); box()

# Add legend using fields package
library(fields)  # For image.plot with legend
image.plot(z0, col=cols, legend.only=TRUE, add=TRUE)

# Panels 2-4: Final states with timing (M=120)
for(nm in precs) {
	zf <- matrix(res_M120[[nm]]$trace[,I], M, M)
	time_mins <- res_M120[[nm]]$time/60
	image(zf, main=sprintf("%s (%.2f mins)", nm, time_mins), cex.lab=1.5,
	      col=cols, xlab=expression(s[x]), ylab=expression(s[y]), axes=FALSE)
	axis(1); axis(2); box()
	image.plot(zf, col=cols, legend.only=TRUE, add=TRUE)
}
dev.off()

# Timing plot across different M sizes
pdf("mala_timing_figure_3.pdf", width=8, height=6)

n_sizes <- M_sizes^2

# Collect all timing values (unique + sorted)
y_ticks <- sort(unique(as.vector(timing_results)))

n_sizes <- M_sizes^2
matplot(n_sizes, timing_results, type="b",
        lty=1:5, pch=c(21,22,24),
        col=1:5, lwd=2,
        log="y",
        xaxt="n",  # 👈 turn off automatic x-axis
        yaxt="n",  # 👈 turn off automatic y-axis
        xlab="Problem Size (n)",
        ylab="Time (seconds, log scale)",
        main="MALA Computation Time vs Problem Size")
# exact x-axis ticks
axis(1, at=n_sizes, labels=n_sizes)
# Add exact timing values on y-axis
axis(2, at=y_ticks, labels=signif(y_ticks, 3), las=1)

legend("topleft", precs, col=1:5, lty=1:5, pch=c(21,22,24), lwd=2)
grid(TRUE, lty=2, col="gray90")

dev.off()

# Compute speedup
speedup_double <- timing_results[,"R-Double"] / timing_results[,"MPCR-Double-CPU"]
speedup_single <- timing_results[,"R-Double"] / timing_results[,"MPCR-Single-CPU"]
speedup_double_gpu <- timing_results[,"R-Double"] / timing_results[,"MPCR-Double-GPU"]
speedup_single_gpu <- timing_results[,"R-Double"] / timing_results[,"MPCR-Single-GPU"]

cat("\nSpeedup (R-Double / MPCR):\n")
cat("MPCR-Double-CPU:", sprintf("%.1fx", speedup_double), "\n")
cat("MPCR-Single-CPU:", sprintf("%.1fx", speedup_single), "\n")
cat("MPCR-Double-GPU:", sprintf("%.1fx", speedup_double_gpu), "\n")
cat("MPCR-Single-GPU:", sprintf("%.1fx", speedup_single_gpu), "\n")

# Save results
save(res_M120, timing_results, speedup_double, speedup_single, speedup_double_gpu, speedup_single_gpu, 
     M_sizes, precs, file="section6.1.RData")
cat("\nResults saved to section6.1.RData\n")

