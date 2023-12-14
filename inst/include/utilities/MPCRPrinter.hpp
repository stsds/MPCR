
#ifndef MPCR_MPCRPRINTER_HPP
#define MPCR_MPCRPRINTER_HPP


#include <Rcpp.h>
#include <sstream>


#ifdef RUNNING_CPP
#define MPCR_PRINTER(message) \
    std::cout<<(message);         \

#else
#define MPCR_PRINTER(message) \
    Rcpp::Rcout<<(message);\

#endif


#endif //MPCR_MPCRPRINTER_HPP
