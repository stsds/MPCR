
WORKDIR <- Sys.getenv("MPR_PACKAGE_DIR")
install.packages(WORKDIR, repos = NULL, type="source")