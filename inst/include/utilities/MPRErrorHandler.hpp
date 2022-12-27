

#ifndef MPR_MPRERRORHANDLER_HPP
#define MPR_MPRERRORHANDLER_HPP

#include <Rcpp.h>
#include <sstream>

/** MPR API Exceptions Macro to use for Errors **/
#define MPR_API_EXCEPTION(MESSAGE, ERROR_CODE) \
MPIAPIException(MESSAGE, __FILE__, __LINE__, __FUNCTION__,true,ERROR_CODE)

/** MPR API Warning Macro to use for Warnings **/
#define MPR_API_WARN(MESSAGE, ERROR_CODE) \
MPIAPIException(MESSAGE, __FILE__, __LINE__, __FUNCTION__,false,ERROR_CODE)

class MPIAPIException {

public:

    MPIAPIException(const char *apMessage,
                    const char *apFileName,
                    int aLineNumber,
                    const char *apFunctionName,
                    bool aIsError,
                    int aErrorCode = 0) {
        std::stringstream ss;

        ss << apMessage << std::endl;

        ss << std::left << std::setfill(' ') << std::setw(10)
           << "File" << ": ";
        ss << std::left << std::setfill(' ') << std::setw(10)
           << apFileName << std::endl;

        ss << std::left << std::setfill(' ') << std::setw(10)
           << "Line" << ": ";
        ss << std::left << std::setfill(' ') << std::setw(10)
           << aLineNumber << std::endl;

        ss << std::left << std::setfill(' ') << std::setw(10)
           << "Function" << ": ";
        ss << std::left << std::setfill(' ') << std::setw(10)
           << apFunctionName << std::endl;

        if (aErrorCode != 0) {
            ss << std::left << std::setfill(' ') << std::setw(10)
               << "Error Code" << ": ";
            ss << std::left << std::setfill(' ') << std::setw(10)
               << aErrorCode << std::endl;
        }


        if (aIsError) {
            MPIAPIException::ThrowError(ss.str());
        } else {
            MPIAPIException::ThrowWarning(ss.str());
        }
    }


    ~MPIAPIException() = default;

private:

    static void
    ThrowError(std::string aString) {
        Rcpp::stop(aString);
    }


    static void
    ThrowWarning(std::string aString) {
        Rcpp::warning(aString);
    }

};


#endif //MPR_MPRERRORHANDLER_HPP
