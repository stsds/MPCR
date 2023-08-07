/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MMPR is an R package provided by the STSDS group at KAUST
 *
 **/

#include <Rcpp.h>


using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif


RcppExport SEXP _rcpp_module_boot_MMPR();

static const R_CallMethodDef CallEntries[] = {
    {"_rcpp_module_boot_MMPR", (DL_FUNC) &_rcpp_module_boot_MMPR, 0},
    {"_rcpp_module_boot_MMPRTile", (DL_FUNC) &_rcpp_module_boot_MMPRTile, 0},
    {NULL, NULL,                                                0}
};

RcppExport void R_init_MMPR(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
