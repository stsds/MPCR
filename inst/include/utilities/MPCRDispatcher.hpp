/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_MPRDISPATCHER_HPP
#define MPCR_MPRDISPATCHER_HPP

#include <common/Definitions.hpp>
#include <utilities/MPCRErrorHandler.hpp>
#include <utilities/FloatingPointHandler.hpp>


using namespace mpcr::definitions;

/** Dispatcher to support Dispatching of template functions with Rcpp
 * Only 10 arguments are supported here.
 * It can be expanded to whatever number of arguments needed
 **/

#define FIRST(...) FIRST_HELPER(__VA_ARGS__, throwaway)
#define FIRST_HELPER(first, ...) first
/**
 * if there's only one argument, expands to nothing.  if there is more
 * than one argument, expands to a comma followed by everything but
 * the first argument.  only supports up to 9 arguments but can be
 * trivially expanded.
 */

#define REST(...) REST_HELPER(NUM(__VA_ARGS__), __VA_ARGS__)
#define REST_HELPER(qty, ...) REST_HELPER2(qty, __VA_ARGS__)
#define REST_HELPER2(qty, ...) REST_HELPER_##qty(__VA_ARGS__)
#define REST_HELPER_ONE(first)
#define REST_HELPER_TWOORMORE(first, ...) , __VA_ARGS__
#define NUM(...) \
    SELECT_10TH(__VA_ARGS__, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE,\
                TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, ONE, throwaway)
#define SELECT_10TH(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, ...) a10

/** Dispatcher for one template arguments **/
#define SIMPLE_DISPATCH(PRECISION, __FUN__, ...)                               \
          switch(PRECISION){                                                   \
              case FLOAT: {                                                    \
               __FUN__<float>(FIRST(__VA_ARGS__)REST(__VA_ARGS__))  ;          \
               break;                                                          \
               }                                                               \
               case DOUBLE: {                                                  \
               __FUN__<double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__))  ;         \
               break;                                                          \
               }                                                               \
               default : {                                                     \
                MPCR_API_EXCEPTION("C++ Error : Type Undefined Dispatcher",     \
                                 (int)PRECISION);                              \
               }                                                               \
          };                                                                   \



#ifdef USE_CUDA

/** Dispatcher for one template arguments **/
#define SIMPLE_DISPATCH_WITH_HALF(PRECISION, __FUN__, ...)                     \
          switch(PRECISION){                                                   \
                case HALF: {                                                   \
               __FUN__<float16>(FIRST(__VA_ARGS__)REST(__VA_ARGS__))  ;        \
               break;                                                          \
               }                                                               \
               case FLOAT: {                                                   \
               __FUN__<float>(FIRST(__VA_ARGS__)REST(__VA_ARGS__))  ;          \
               break;                                                          \
               }                                                               \
               case DOUBLE: {                                                  \
               __FUN__<double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__))  ;         \
               break;                                                          \
               }                                                               \
               default : {                                                     \
                MPCR_API_EXCEPTION("C++ Error : Type Undefined Dispatcher",    \
                                 (int)PRECISION);                              \
               }                                                               \
          };                                                                   \


/** Instantiators for Template functions with a given return type
 * (One template argument)
 **/
#define SIMPLE_INSTANTIATE_WITH_HALF(RETURNTYPE, __FUN__, ...) \
        template RETURNTYPE __FUN__<float16> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        SIMPLE_INSTANTIATE(RETURNTYPE, __FUN__, FIRST(__VA_ARGS__)REST(__VA_ARGS__))

#else

/** Dispatcher for one template arguments **/
#define SIMPLE_DISPATCH_WITH_HALF(PRECISION, __FUN__, ...)                     \
        SIMPLE_DISPATCH(PRECISION, __FUN__, __VA_ARGS__)

#define SIMPLE_INSTANTIATE_WITH_HALF(RETURNTYPE, __FUN__, ...) \
        SIMPLE_INSTANTIATE(RETURNTYPE, __FUN__, FIRST(__VA_ARGS__)REST(__VA_ARGS__))

#endif

/** Dispatcher for three template arguments **/
#define DISPATCHER(PRECISION, __FUN__, ...)                                    \
          switch(PRECISION){                                                   \
               case DFD: {                                                     \
               __FUN__<double,float,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));\
               break;                                                          \
               }                                                               \
               case FDD: {                                                     \
               __FUN__<float,double,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));\
               break;                                                          \
               }                                                               \
               case FFF: {                                                     \
               __FUN__<float,float,float>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));\
               break;                                                          \
               }                                                               \
               case DDD: {                                                     \
               __FUN__<double,double,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));\
               break;                                                          \
               }                                                               \
               default : {                                                     \
               MPCR_API_EXCEPTION("C++ Error : Type Undefined Dispatcher",      \
                                (int)PRECISION);                               \
               }                                                               \
          }  ;                                                                 \

/** Instantiators for Template functions with a given return type
 * (One template argument)
 **/
#define SIMPLE_INSTANTIATE(RETURNTYPE, __FUN__, ...) \
        template RETURNTYPE __FUN__<float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;  \




/** Instantiators for Template functions with a given return type
 * (Three template argument)
 **/
#define INSTANTIATE(RETURNTYPE, __FUN__, ...) \
        template RETURNTYPE __FUN__<double,float,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<float,double,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float,float,float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<double,double,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\


/** Instantiators for Template functions with a given return type
 * (Three template argument)
 **/
#define COPY_INSTANTIATE(RETURNTYPE, __FUN__, ...) \
        template RETURNTYPE __FUN__<float16,float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float16,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<float,float16> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<double,float16> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<double,float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;  \
        template RETURNTYPE __FUN__<float,float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;  \
        template RETURNTYPE __FUN__<double,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;  \
        template RETURNTYPE __FUN__<float16,float16> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;  \
        template RETURNTYPE __FUN__<int64_t,float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;  \
        template RETURNTYPE __FUN__<int64_t,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;  \


/** Instantiators for Template functions with a given return type
 * (Three template argument)
 **/
#define COPY_INSTANTIATE_ONE(RETURNTYPE, __FUN__, ...) \
        template RETURNTYPE __FUN__<float16,float> (const float16*,float*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float16,double> (const float16*,double*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<float,float16> (const float*,float16*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float,double> (const float*,double*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<double,float16> (const double*,float16*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<double,float> (const double*,float*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;   \
        template RETURNTYPE __FUN__<double,double> (const double*,double*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float,float> (const float*,float*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;     \
        template RETURNTYPE __FUN__<float16,float16> (const float16*,float16*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<int64_t,float> (const int64_t*,float*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<int64_t,double> (const int64_t*,double*,FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \


/**
 * @def MPCR_INSTANTIATE_CLASS
 * @brief Macro definition to instantiate the MPCR template classes with supported types.
**/

#define MPCR_INSTANTIATE_CLASS(TEMPLATE_CLASS)   template class TEMPLATE_CLASS<float>;  \
                                                    template class TEMPLATE_CLASS<double>;


#define CONCATENATE(a, b) a ## b

#define CONCATENATE3(a, b, c) a ## b ## c

// Macro to define the function name based on precision
#define CONCATENATE_FUNCTION_NAME(NAME_ONE, PRECISION, NAME_TWO) CONCATENATE3(NAME_ONE, PRECISION, NAME_TWO)

#define CALL_FUNCTION_S(NAME_ONE, NAME_TWO, ...) CONCATENATE_FUNCTION_NAME(NAME_ONE, S, NAME_TWO)(__VA_ARGS__)
#define CALL_FUNCTION_D(NAME_ONE, NAME_TWO, ...) CONCATENATE_FUNCTION_NAME(NAME_ONE, D, NAME_TWO)(__VA_ARGS__)
#define CALL_FUNCTION_H(NAME_ONE, NAME_TWO, ...) CONCATENATE_FUNCTION_NAME(NAME_ONE, H, NAME_TWO)(__VA_ARGS__)


/** Dispatcher for CUDA functions **/
#define CUDA_FUNCTIONS_NAME_DISPATCHER(NAME_ONE, NAME_TWO, ...)                 \
        if constexpr(is_float<T>()){                                           \
               CALL_FUNCTION_S(NAME_ONE, NAME_TWO, __VA_ARGS__);               \
        }else if constexpr(is_double<T>()) {                                   \
               CALL_FUNCTION_D(NAME_ONE, NAME_TWO, __VA_ARGS__)  ;             \
        }else if constexpr(is_half<T>()){                                      \
               CALL_FUNCTION_H(NAME_ONE, NAME_TWO, __VA_ARGS__)  ;             \
        }                                                                      \


#endif //MPCR_MPRDISPATCHER_HPP
