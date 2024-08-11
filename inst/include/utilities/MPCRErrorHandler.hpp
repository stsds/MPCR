/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_MPRERRORHANDLER_HPP
#define MPCR_MPRERRORHANDLER_HPP


#include <utilities/MPCRPrinter.hpp>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif


/** MPCR API Exceptions Macro to use for Errors **/
#define MPCR_API_EXCEPTION(MESSAGE, ERROR_CODE) \
MPCRAPIException(MESSAGE, __FILE__, __LINE__, __FUNCTION__,true,ERROR_CODE)

/** MPCR API Warning Macro to use for Warnings **/
#define MPCR_API_WARN(MESSAGE, WARNING_CODE) \
MPCRAPIException(MESSAGE, __FILE__, __LINE__, __FUNCTION__,false,WARNING_CODE)


#ifdef USE_CUDA
/**
 * @brief
 * Useful macro wrapper for all cuda API calls to ensure correct returns,
 * and error throwing on failures.
 */
#define GPU_ERROR_CHECK(ans) { MPCRAPIException::AssertGPU((ans), __FILE__, __LINE__); }

#endif

class MPCRAPIException {

public:

    MPCRAPIException(const char *apMessage,
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
            MPCRAPIException::ThrowError(ss.str());
        } else {
            MPCRAPIException::ThrowWarning(ss.str());
        }
    }


    ~MPCRAPIException() = default;

#ifdef USE_CUDA
    /**
     * @brief
     * Function to assert the return code of a CUDA API call, and ensure
     * it completed successfully.
     *
     * @param[in] aCode
     * The code returned from the CUDA API call.
     *
     * @param[in] aFile
     * The name of the file that the assertion was called from.
     *
     * @param[in] aLine
     * The line number in the file that the assertion that was called from.
     *
     * @param[in] aAbort
     * Boolean to indicate whether to exit on failure or not, by default is true and will exit.
     */
    inline static void AssertGPU(cudaError_t aCode, const char *aFile, int aLine, bool aAbort=true)
    {
        if (aCode != cudaSuccess)
        {
#ifdef RUNNING_CPP
            char s[200];
            sprintf((char*)s,"GPU Assert: %s %s %d\n", cudaGetErrorString(aCode), aFile, aLine);
            throw std::invalid_argument(s);
#else
            std::string s="GPU Assert:  "+std::string(cudaGetErrorString(aCode));
            Rcpp::stop(s);
#endif
        }
    }
#endif


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


#endif //MPCR_MPRERRORHANDLER_HPP
