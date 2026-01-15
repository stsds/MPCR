cat("\nRunning Codes for Section 6.1\n")
cat("\nTakes about \n")

library(MPCR)

M <- 150; n <- M^2 # Setup: spatial grid
locs <- expand.grid(x=(0:(M-1))/(M-1), y=(0:(M-1))/(M-1))
D <- as.matrix(dist(locs))

# Target distribution: mean=0, covariance=exp(-D/0.5)
mu <- rep(0, n); sig <- exp(-D/0.5)
# Preconditioning matrix for MALA
pre_M <- exp(-D/0.05); h <- 0.01; I <- 2000

# Function to run MALA with different precisions
run_mala <- function(prec='R-Double', placement='CPU') {
	# Convert to MPCR if needed (consistent precision throughout)
	use_mpcr <- grepl('MPCR', prec)
	op_place <- "CPU"
	if (use_mpcr) {

		# Precision
		p <- if (prec == "MPCR-Single-CPU" || prec == "MPCR-Single-GPU") "single" else "double"

		# Normalize placement to expected values
		op_place  <- if (prec == "MPCR-Double-CPU" || prec == "MPCR-Double-CPU") "CPU" else "GPU"

		# Set MPCR operation placement once
		MPCR.SetOperationPlacement(placement = op_place)
		print(op_place)
		# Convert once (pass placement only when GPU is requested)
		if (op_place == "GPU") {
			mu    <- as.MPCR(mu,    n, 1, p, placement = "GPU")
			sig   <- as.MPCR(sig,   n, n, p, placement = "GPU")
			pre_M <- as.MPCR(pre_M, n, n, p, placement = "GPU")
		} else {
			mu    <- as.MPCR(mu,    n, 1, p)
			sig   <- as.MPCR(sig,   n, n, p)
			pre_M <- as.MPCR(pre_M, n, n, p)
		}

	}
	print(op_place)
	# Precompute matrices
	# MPCR.SetOperationPlacement(placement = op_place)
	sig.inv <- solve(sig)
	L <- t(chol(if(use_mpcr) pre_M$PerformMult(h) else h*pre_M))
	pre_M_inv <- solve(if(use_mpcr) pre_M$PerformMult(h) else h*pre_M)

	# Gradient and gradient step
	grad <- function(x) sig.inv %*% x
	step <- function(x, t) {
		if(use_mpcr) x - (pre_M %*% grad(x))$PerformMult(t)
		else x - t * pre_M %*% grad(x)
	}

	# Initialize chain
	set.seed(1234); z <- runif(n)
	if(use_mpcr) z <- as.MPCR(z, n, 1, p, placement=op_place)  # Use consistent precision
	trace <- matrix(NA, n, I)

	# MALA iterations
	t0 <- Sys.time()
	for(i in 1:I) {
		MPCR.SetOperationPlacement(placement = op_place)
		set.seed(i)
		eps <- if(use_mpcr) as.MPCR(rnorm(n), n, 1, p, placement=op_place) else rnorm(n)
		z_prop <- step(z, 0.5*h) + L %*% eps

		# Compute log ratio (helper for MPCR conversion)
		num <- function(x) if(use_mpcr) MPCR.ToNumericVector(x) else as.numeric(x)
		p_prop <- -0.5 * num(t(z_prop-mu) %*% sig.inv %*% (z_prop-mu))
		p_curr <- -0.5 * num(t(z-mu) %*% sig.inv %*% (z-mu))
		q_curr <- -0.5 * num(t(z-step(z_prop,0.5*h)) %*% pre_M_inv %*% 
				     (z-step(z_prop,0.5*h)))
		q_prop <- -0.5 * num(t(z_prop-step(z,0.5*h)) %*% pre_M_inv %*% 
				     (z_prop-step(z,0.5*h)))
		#cat("2. After creation - Operation placement:", MPCR.GetOperationPlacement(), "\n")
		if(runif(1) < exp(min(0, p_prop - p_curr + q_curr - q_prop)))
			z <- z_prop
		trace[,i] <- if(use_mpcr) MPCR.ToNumericVector(z) else z
	}
	list(trace=trace, time=difftime(Sys.time(), t0, units="mins"))
}

# Run all precisions and create figure
res <- lapply(c('R-Double','MPCR-Double-CPU','MPCR-Single-CPU', 'MPCR-Double-GPU','MPCR-Single-GPU'), run_mala)
names(res) <- c('R-Double','MPCR-Double-CPU','MPCR-Single-CPU', 'MPCR-Double-GPU','MPCR-Single-GPU')


#res <- lapply(c('MPCR-Single-GPU'), run_mala)
#names(res) <- c('MPCR-Single-GPU')

# Visualization 
pdf("mh_results.pdf", width=16, height=4)
par(mfrow=c(2,3), mar=c(5,5,3,4)) 
cols <- rev(rainbow(100, start=0, end=4/6)) 
# Panel 1: Initial z0
set.seed(1234); z0 <- matrix(runif(n), M, M)
image(z0, main=expression(bold(z[0])), col=cols, cex.lab=1.5,
      xlab=expression(s[x]), ylab=expression(s[y]), axes=FALSE)
axis(1); axis(2); box()
# Add legend using fields package
library(fields)  # For image.plot with legend
image.plot(z0, col=cols, legend.only=TRUE, add=TRUE)
# Panels 2-4: Final states with timing
for(nm in names(res)) {
	zf <- matrix(res[[nm]]$trace[,I], M, M)
	image(zf, main=sprintf("%s (%.2f mins)", nm, res[[nm]]$time), cex.lab=1.5,
	      col=cols, xlab=expression(s[x]), ylab=expression(s[y]), axes=FALSE)
	axis(1); axis(2); box()
	image.plot(zf, col=cols, legend.only=TRUE, add=TRUE)
}
dev.off()

cat("\nDONE...\n")
