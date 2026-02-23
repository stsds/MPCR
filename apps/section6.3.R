cat("\nRunning Codes for Section 6.3\n")
#cat("\nTakes about 18 minutes\n")

library(MPCR)
library(ncdf4)
library(fields)
library(maps)
# Load climate data (download from Google Drive link)
link <- "b.e21.BHISTcmip6.f09_g17.LE2-1001.001.cam.h3.UBOT.2010010100-2014123100.nc"
nc <- nc_open(link)
lat <- ncvar_get(nc, "lat"); lon <- (ncvar_get(nc, "lon") + 180) %% 360 - 180
# Extract North Atlantic region
lat_idx <- which(lat >= 7 & lat <= 85)
lon_idx <- which(lon >= -180 & lon <= -20)
ubot_raw <- ncvar_get(nc, "UBOT", start=c(lon_idx[1], lat_idx[1], 1),
                      count=c(length(lon_idx), length(lat_idx), -1))
# Reshape to matrix (time x space)
valid <- which(!is.na(ubot_raw[,,1]))
ubot <- matrix(0, dim(ubot_raw)[3], length(valid))
for(i in 1:dim(ubot_raw)[3]) ubot[i,] <- ubot_raw[,,i][valid]

# Function to run PCA with different precisions
run_pca <- function(prec='R-Double') {
  use_mpcr <- grepl('MPCR', prec)
  t0 <- Sys.time()

  if(use_mpcr) {

	  # Precision
	  p <- if (prec == "MPCR-Single-CPU" || prec == "MPCR-Single-GPU") "single" else "double"

	  # Normalize placement to expected values
	  op_place  <- if (prec == "MPCR-Single-CPU" || prec == "MPCR-Double-CPU") "CPU" else "GPU"

	  # Set MPCR operation placement once
	  MPCR.SetOperationPlacement(placement = op_place)
	  print(op_place)
	  X <- as.MPCR(ubot, nrow(ubot), ncol(ubot), p, placement=op_place)


	  s <- svd(X)
	  list(u=MPCR.ToNumericMatrix(s$u), v=MPCR.ToNumericMatrix(s$v),
	       d=MPCR.ToNumericVector(s$d), time=difftime(Sys.time(), t0, units="mins"))
  } else {
	  s <- svd(ubot)
	  list(u=s$u, v=s$v, d=s$d, time=difftime(Sys.time(), t0, units="mins"))
  }
}

# Run all precisions
#res <- lapply(c('R-Double','MPCR-Double-CPU','MPCR-Single-CPU'), run_pca)
#names(res) <- c('R-Double','MPCR-Double-CPU','MPCR-Single-CPU')

res <- lapply(c('R-Double','MPCR-Double-CPU','MPCR-Single-CPU', 'MPCR-Double-GPU','MPCR-Single-GPU'), run_pca)
names(res) <- c('R-Double','MPCR-Double-CPU','MPCR-Single-CPU', 'MPCR-Double-GPU','MPCR-Single-GPU')


#res <- lapply(c('MPCR-Double-CPU','MPCR-Double-CPU','MPCR-Single-CPU', 'MPCR-Double-GPU','MPCR-Single-GPU'), run_pca)
#names(res) <- c('MPCR-Double-CPU','MPCR-Double-CPU','MPCR-Single-CPU', 'MPCR-Double-GPU','MPCR-Single-GPU')


# Visualizations
library(RColorBrewer)
# (1) Eigenvalues
pdf("pca_eigenvalues.pdf", width=8, height=6)
pct_var <- lapply(res, function(r) 100*r$d^2/sum(r$d^2))
matplot(1:20, sapply(pct_var, "[", 1:20), type="b", lty=1:3, 
	pch=c(21,22,24), col=1:5, lwd=2, ylim=c(0,40),
	xlab="Eigenvalue", ylab="Percentage of variance [%]",
	main="Percent of Explained Variance")
legend("topright", names(res), col=1:5, lty=1:5, pch=c(21,22,24), lwd=2)
dev.off()


# (2) EOFs (5x3 grid)
pdf("pca_eofs.pdf", width = 30, height = 15)

par(mfrow = c(3, 5), mar = c(4, 7, 3, 7))
cols <- colorRampPalette(brewer.pal(9, "BrBG"))(100)

for (i in 1:5) {
  for (nm in names(res)) {

    Z <- matrix(NA, length(lon_idx), length(lat_idx))
    Z[valid] <- if (is.matrix(res[[nm]]$v))
      res[[nm]]$v[, i]
    else
      matrix(res[[nm]]$v, nrow = length(valid))[, i]

    eof_label <- paste0(i, c("st", "nd", "rd", "th", "th")[i], " EOF")
    title_text <- ifelse(
      i == 1,
      sprintf("%s (%.2f mins)", nm, as.numeric(res[[nm]]$time)),
      ""
    )

    image.plot(
      lon[lon_idx], lat[lat_idx], Z,
      col = cols,
      main = title_text,
      xlab = if (i == 5) "Longitude" else "",
      ylab = if (nm == names(res)[1]) "Latitude" else "",
      cex.main = 2.5,
      cex.lab  = 2
    )

    if (nm == names(res)[1])
      mtext(eof_label, side = 2, line = 5, cex = 1.2, font = 2)

    lines(map("world", plot = FALSE))
    box()
  }
}

dev.off()
cat("\nDONE...\n")

