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
              case HALF: {                                                     \
              __FUN__<float>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));             \
              break;                                                           \
              }                                                                \
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

/** Dispatcher for three template arguments **/
#define DISPATCHER(PRECISION, __FUN__, ...)                                    \
          switch(PRECISION){                                                   \
               case FSF: {                                                     \
               __FUN__<float,float16,float>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));  \
               break;                                                          \
               }                                                               \
               case SFF: {                                                     \
               __FUN__<float16,float,float>(FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
               break;                                                          \
               }                                                               \
               case DSD: {                                                     \
               __FUN__<double,float16,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));\
               break;                                                          \
               }                                                               \
               case SDD: {                                                     \
               __FUN__<float16,double,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));\
               break;                                                          \
               }                                                               \
               case DFD: {                                                     \
               __FUN__<double,float,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));\
               break;                                                          \
               }                                                               \
               case FDD: {                                                     \
               __FUN__<float,double,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));\
               break;                                                          \
               }                                                               \
               case SSS: {                                                     \
               __FUN__<float16,float16,float16>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));      \
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
               case SSD: {                                                     \
               __FUN__<float16,float16,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));   \
               break;                                                          \
               }                                                               \
               case SSF: {                                                     \
               __FUN__<float16,float16,float>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));    \
               break;                                                          \
               }                                                               \
               case FFD: {                                                     \
               __FUN__<float,float,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__));\
               break;                                                          \
               }                                                               \
               case SFD: {                                                     \
               __FUN__<float16,float,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__)); \
               break;                                                          \
               }                                                               \
               case FSD: {                                                     \
               __FUN__<float,float16,double>(FIRST(__VA_ARGS__)REST(__VA_ARGS__)); \
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
        template RETURNTYPE __FUN__<float16> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;  \



/** Instantiators for Template functions with a given return type
 * (Three template argument)
 **/
#define INSTANTIATE(RETURNTYPE, __FUN__, ...) \
        template RETURNTYPE __FUN__<float,float16,float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float16,float,float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<double,float16,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float16,double,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<double,float,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<float,double,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float16,float16,float16> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float,float,float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<double,double,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<float16,float16,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float16,float16,float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<float,float,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float16,float,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float,float16,double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\


#define FLOATING_POINT_INST(RETURNTYPE, __FUN__, ...) \
        template RETURNTYPE __FUN__<float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \


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

#endif //MPCR_MPRDISPATCHER_HPP
