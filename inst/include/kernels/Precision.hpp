/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_PRECISION_HPP
#define MPCR_PRECISION_HPP

#include <utilities/FloatingPointHandler.hpp>
#include <utilities/MPCRErrorHandler.hpp>
#include <common/Definitions.hpp>


using namespace mpcr::definitions;

namespace mpcr {
    namespace precision {

        /**
         * @brief
         * Get Precision of output element (to promote output in case of
         * multi-precision operations)
         *
         * @param[in] aPrecisionA
         * Precision of the first input
         * @param[in] aPrecisionB
         * Precision of the second input
         *
         * @returns
         * Precision of the output objects
         */
        inline
        Precision
        GetOutputPrecision(const Precision &aPrecisionA,
                           const Precision &aPrecisionB) {
            if (aPrecisionA > 3 || aPrecisionB > 3) {
                MPCR_API_EXCEPTION("Unknown Type Value", -1);
            }
            return ( aPrecisionA >= aPrecisionB ) ? aPrecisionA : aPrecisionB;
        }


        /**
         * @brief
         * Get Operation Order according to input,input,output for Dispatcher
         *
         * @param[in] aPrecisionA
         * Precision of the first input
         * @param[in] aPrecisionB
         * Precision of the second input
         * @param[in] aPrecisionC
         * Precision of the output
         *
         * @returns
         * Operation order used by the Dispatcher
         */
        inline
        Precision
        GetOperationPrecision(const Precision &aPrecisionA,
                              const Precision &aPrecisionB,
                              const Precision &aPrecisionC) {

            /** this formula is used instead of writing many if/else cases **/

            /** each precision is multiplied by a prime number according to its
             *  position ( Generating a unique value for each operation)
             **/
            int temp =
                ( 3 * aPrecisionA ) + ( 5 * aPrecisionB ) + ( 7 * aPrecisionC );

            auto operation = static_cast<Precision>(temp);
            return operation;
        }


        /**
         * @brief
         * Get Precision for any MPCR Class to make sure no Input is initialized
         * as anything other than the supported 3 precisions. (Int Version)
         *
         * @param[in] aPrecision
         * int describing required precision
         *
         * @returns
         * Precision out the 3-supported precision ,or throw exception in-case
         * it's not supported
         */
        inline
        Precision
        GetInputPrecision(const int &aPrecision) {
            if (aPrecision > 0 && aPrecision < 4) {
                if (aPrecision == 0 && !USING_HALF) {
                    MPCR_API_WARN(
                        "Your Compiler doesn't support 16-Bit ,32-Bit will be used",
                        1);
                    return FLOAT;
                }
                return static_cast<Precision>(aPrecision);
            } else {
                MPCR_API_EXCEPTION(
                    "Error in Initialization : Unknown Type Value",
                    aPrecision);
            }
            return ERROR;
        }


        /**
         * @brief
         * Get Precision for any MPCR Class to make sure no Input is initialized
         * as anything other than the supported 3 precisions. (String Version).
         * Transforms the string to lower case to ensure proper initialization
         *
         * @param[in] aPrecision
         * int describing required precision
         *
         * @returns
         * Precision out the 3-supported precision ,or throw exception in-case
         * it's not supported
         */
        inline
        Precision
        GetInputPrecision(std::string aPrecision) {
            std::transform(aPrecision.begin(), aPrecision.end(),
                           aPrecision.begin(), ::tolower);

            if (aPrecision == "float" || aPrecision == "single") {
                return FLOAT;
            } else if (aPrecision == "double") {
                return DOUBLE;
            } else if (aPrecision == "half") {
                if constexpr(!USING_HALF) {
                    MPCR_API_WARN(
                        "Your Compiler doesn't support 16-Bit ,32-Bit will be used",
                        1);
                    return FLOAT;
                } else {
                    return HALF;
                }
            } else {
                auto msg = "Error in Initialization : Unknown Type Value" +
                           std::string(aPrecision);
                MPCR_API_EXCEPTION(
                    msg.c_str(), -1
                );
            }
            return ERROR;
        }


        /**
         * @brief
         * Get Precision for any Precision class as a string
         *
         * @param[in] aPrecision
         * Precision enum
         *
         * @returns
         * String describing the enum.
         */
        inline
        std::string
        GetPrecisionAsString(const Precision &aPrecision) {

            if (aPrecision == HALF) {
                return "16-Bit";
            } else if (aPrecision == FLOAT) {
                return "32-Bit";
            } else if (aPrecision == DOUBLE) {
                return "64-Bit";
            } else {
                MPCR_API_EXCEPTION(
                    "Error in Initialization : Unknown Type Value",
                    (int) aPrecision);
            }
            return "Unknown Type";
        }


    }

}


#endif //MPCR_PRECISION_HPP
