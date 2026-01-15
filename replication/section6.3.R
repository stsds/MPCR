cat("\nRunning Codes for Section 6.3\n")
cat("\nTakes about 40 minutes\n")

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
    p <- if(prec == 'MPCR-Single') 'single' else 'double'
    X <- as.MPCR(ubot, nrow(ubot), ncol(ubot), p)
    s <- svd(X)
    list(u=MPCR.ToNumericMatrix(s$u), v=MPCR.ToNumericMatrix(s$v),
         d=MPCR.ToNumericVector(s$d), time=difftime(Sys.time(), t0, units="mins"))
  } else {
    s <- svd(ubot)
    list(u=s$u, v=s$v, d=s$d, time=difftime(Sys.time(), t0, units="mins"))
  }
}

# Run all precisions
res <- lapply(c('R-Double','MPCR-Double','MPCR-Single'), run_pca)
names(res) <- c('R-Double','MPCR-Double','MPCR-Single')

# Visualizations
library(RColorBrewer)
# (1) Eigenvalues
pdf("pca_eigenvalues.pdf", width=8, height=6)
pct_var <- lapply(res, function(r) 100*r$d^2/sum(r$d^2))
matplot(1:20, sapply(pct_var, "[", 1:20), type="b", lty=1:3, 
        pch=c(21,22,24), col=1:3, lwd=2, ylim=c(0,40),
        xlab="Eigenvalue", ylab="Percentage of variance [%]",
        main="Percent of Explained Variance")
legend("topright", names(res), col=1:3, lty=1:3, pch=c(21,22,24), lwd=2)
dev.off()

# (2) EOFs (3x3 grid)
pdf("pca_eofs.pdf", width=14, height=8)
par(mfrow=c(3,3), mar=c(4,7,3,7))
cols <- colorRampPalette(brewer.pal(9, "BrBG"))(100)
for(i in 1:3) for(nm in names(res)) {
  Z <- matrix(NA, length(lon_idx), length(lat_idx))
  Z[valid] <- if(is.matrix(res[[nm]]$v)) res[[nm]]$v[,i] else 
    matrix(res[[nm]]$v, nrow=length(valid))[,i]
  eof_label <- c("1st EOF", "2nd EOF", "3rd EOF")[i]
  title_text <- ifelse(i == 1, sprintf("%s (%.2f mins)", nm, as.numeric(res[[nm]]$time)), "")
  image.plot(lon[lon_idx], lat[lat_idx], Z, col=cols,
             main=title_text,
             xlab=if(i==3) "Longitude" else "", 
             ylab=if(nm=="R-Double") "Latitude" else "",
             cex.main=2.5, cex.lab=2)
  if(nm=="R-Double") mtext(eof_label, side=2, line=5, cex=1.2, font=2)
  lines(map("world", plot=FALSE)); box()
}
dev.off()

cat("\nDONE...\n")


