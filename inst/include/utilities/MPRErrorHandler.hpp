/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MMPR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPR_MPRERRORHANDLER_HPP
#define MPR_MPRERRORHANDLER_HPP

#include <Rcpp.h>
#include <sstream>

/** MPR API Exceptions Macro to use for Errors **/
#define MPR_API_EXCEPTION(MESSAGE, ERROR_CODE) \
MPRAPIException(MESSAGE, __FILE__, __LINE__, __FUNCTION__,true,ERROR_CODE)

/** MPR API Warning Macro to use for Warnings **/
#define MPR_API_WARN(MESSAGE, WARNING_CODE) \
MPRAPIException(MESSAGE, __FILE__, __LINE__, __FUNCTION__,false,WARNING_CODE)

class MPRAPIException {

public:

    MPRAPIException(const char *apMessage,
                    const char *apFileName,
                    int aLineNumber,
                    const char *apFunctionName,
                    bool aIsError,
                    int aErrorCode = 0) {
        std::stringstream ss;

        ss << apMessage << std::endl;

#ifdef RUNNING_CPP
        ss << std::left << std::setfill(' ') << std::setw(14)
           << "File" << ": ";
        ss << std::left << std::setfill(' ') << std::setw(14)
           << apFileName << std::endl;

        ss << std::left << std::setfill(' ') << std::setw(14)
           << "Line" << ": ";
        ss << std::left << std::setfill(' ') << std::setw(14)
           << aLineNumber << std::endl;
#endif
        ss << std::left << std::setfill(' ') << std::setw(14)
           << "Function" << ": ";
        ss << std::left << std::setfill(' ') << std::setw(14)
           << apFunctionName << std::endl;

        if (aErrorCode != 0 && aIsError) {
            ss << std::left << std::setfill(' ') << std::setw(14)
               << "Error Code" << ": ";
            ss << std::left << std::setfill(' ') << std::setw(14)
               << aErrorCode << std::endl;
        }


        if (aIsError) {
            MPRAPIException::ThrowError(ss.str());
        } else {
            MPRAPIException::ThrowWarning(ss.str());
        }
    }


    ~MPRAPIException() = default;

private:

    static void
    ThrowError(std::string aString) {
#ifdef RUNNING_CPP
        throw std::invalid_argument(aString.c_str());
#endif
#ifndef RUNNING_CPP
        Rcpp::stop(aString);
#endif
    }


    static void
    ThrowWarning(std::string aString) {
#ifndef RUNNING_CPP
        Rcpp::warning(aString);
#endif
    }

};


#endif //MPR_MPRERRORHANDLER_HPP
