library(MPCR)
library(Matrix)

# Global parameters
alpha_true <- 0.6; sigma_true <- 0.1; beta_true <- 10

# Helper to generate data for any n
gen_data <- function(n_size) {
  locs <- cbind(seq(0, n_size, length.out=n_size), 0)
  D <- as.matrix(dist(locs))
  cov <- sigma_true * exp(-D/alpha_true)
  set.seed(4); x <- drop(matrix(rnorm(n_size), 1, n_size) %*% t(chol(cov)))
  p <- 1/(1 + exp(-beta_true*x))
  list(D=D, y=rbinom(n_size, 1, p), n=n_size)
}

# INLA function
run_inla <- function(prec='R-Double', D, y, n) {
  use_mpcr <- grepl('MPCR', prec)
  p <- if(prec=='MPCR-Single') 'single' else 'double'
  alpha_vec <- seq(0.05, 0.95, length=21)
  
  calc_Q <- function(alpha) {
    cov <- sigma_true * exp(-D/alpha)
    if(use_mpcr) cov <- as.MPCR(cov, n, n, p)
    solve(cov)
  }
  
  calc_x0 <- function(alpha, tol=1e-12) {
    Q <- calc_Q(alpha)
    x <- x0 <- rep(0, n)
    repeat {
      g1 <- beta_true*y - beta_true*exp(beta_true*x)/(1+exp(beta_true*x))
      g2 <- -beta_true^2*exp(beta_true*x)/(1+exp(beta_true*x))^2
      diag_g <- bandSparse(n=n, k=0, diagonals=list(g2))
      if(use_mpcr) {
        diag_g <- as.MPCR(as.matrix(diag_g), n, n, p)
        mode <- as.MPCR(g1-x0*g2, n, 1, p)
        x <- MPCR.ToNumericVector(solve(Q-diag_g) %*% mode)
      } else {
        x <- drop(solve(Q-diag_g) %*% (g1-x0*g2))
      }
      if(mean((x-x0)^2) < tol) break else x0 <- x
    }
    x
  }
  
  calc_lpost <- function(alpha) {
    x0 <- calc_x0(alpha)
    Q <- calc_Q(alpha)
    diag_term <- diag(beta_true^2*exp(beta_true*x0)/(1+exp(beta_true*x0))^2, n, n)
    if(use_mpcr) diag_term <- as.MPCR(diag_term, n, n, p)
    H <- Q + diag_term
    chol_Q <- chol(Q); chol_H <- chol(H)
    
    if(use_mpcr) {
      logdet_Q <- log(diag(chol_Q))$Sum() * 2
      logdet_H <- log(diag(chol_H))$Sum() * 2
      x0_mpcr <- as.MPCR(x0, n, 1, p)
      quad <- (chol_Q %*% x0_mpcr)$SquareSum()
    } else {
      logdet_Q <- 2*sum(log(diag(chol_Q)))
      logdet_H <- 2*sum(log(diag(chol_H)))
      quad <- drop(t(x0) %*% Q %*% x0)
    }
    sum(beta_true*x0*y - log1p(exp(beta_true*x0))) + 
      0.5*logdet_Q - 0.5*quad - 0.5*logdet_H
  }
  
  t0 <- Sys.time()
  lpost <- sapply(alpha_vec, calc_lpost)
  time_elapsed <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  
  lpost <- lpost - mean(lpost)
  h <- alpha_vec[2] - alpha_vec[1]
  w <- c(1, rep(c(4,2), (length(alpha_vec)-3)/2), 4, 1)
  Z <- sum(w * exp(lpost)) * h/3
  
  list(alpha=alpha_vec, posterior=exp(lpost)/Z, time=time_elapsed)
}

# Compute for all sizes
sizes <- c(14400)
precs <- c('R-Double', 'MPCR-Double', 'MPCR-Single')
timing_results <- matrix(NA, length(sizes), length(precs))
res <- NULL  # Will store n=2500 results for plotting

cat("Computing INLA for all sizes...\n")
for(i in seq_along(sizes)) {
  cat("\nSize n =", sizes[i], "\n")
  d <- gen_data(sizes[i])
  
  for(j in seq_along(precs)) {
    cat("  ", precs[j], "...")
    result <- run_inla(precs[j], d$D, d$y, d$n)
    timing_results[i,j] <- result$time
    
    # Save n=2500 results for posterior plot
    if(sizes[i] == 2500 && is.null(res)) {
      res <- list()
    }
    if(sizes[i] == 2500) {
      res[[precs[j]]] <- result
    }
    
    cat(" ", round(timing_results[i,j], 2), "secs\n")
  }
}

colnames(timing_results) <- precs
rownames(timing_results) <- paste0("n=", sizes)
cat("\nTiming Results (seconds):\n")
print(timing_results)

# Posterior plot for n=2500
pdf("inla_posterior.pdf", width=10, height=8)
cols <- c("black", "red", "blue"); ltys <- c(1, 2, 4)
plot(res[[1]]$alpha, res[[1]]$posterior, type="l", lwd=2.5, col=cols[1], lty=ltys[1],
     xlab=expression(alpha), ylab="Posterior", ylim=c(0,4.5),
     main=expression(paste("Posterior Distribution of ", alpha, " (n=2500)")))
for(i in 2:3) lines(res[[i]]$alpha, res[[i]]$posterior, lwd=2.5, col=cols[i], lty=ltys[i])
for(i in 1:3) {
  idx <- seq(1, 21, by=3)
  points(res[[i]]$alpha[idx], res[[i]]$posterior[idx], pch=c(16,17,15)[i], col=cols[i], cex=0.8)
}
legend("topright", names(res), col=cols, lwd=2.5, lty=ltys, pch=c(16,17,15), pt.cex=0.8)
abline(v=alpha_true, lty=3, col="gray", lwd=1.5)
dev.off()

# Timing plot
pdf("inla_timing.pdf", width=8, height=6)
matplot(sizes, timing_results, type="b", lty=1:3, pch=c(21,22,24), 
        col=1:3, lwd=2, log="y",
        xlab="Problem Size (n)", ylab="Time (seconds, log scale)",
        main="INLA Computation Time vs Problem Size")
legend("topleft", precs, col=1:3, lty=1:3, pch=c(21,22,24), lwd=2)
grid(TRUE, lty=2, col="gray90")
dev.off()

# Speedup
speedup_double <- timing_results[,"R-Double"] / timing_results[,"MPCR-Double"]
speedup_single <- timing_results[,"R-Double"] / timing_results[,"MPCR-Single"]
cat("\nSpeedup (R-Double / MPCR):\n")
cat("MPCR-Double:", sprintf("%.1fx", speedup_double), "\n")
cat("MPCR-Single:", sprintf("%.1fx", speedup_single), "\n")

# Save all results
save(res, timing_results, speedup_double, speedup_single, sizes, precs, alpha_true,
     file="section6.4.RData")
cat("\nResults saved to section6.4.RData\n")
