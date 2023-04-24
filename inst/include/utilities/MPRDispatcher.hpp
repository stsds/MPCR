
#ifndef MPR_MPRDISPATCHER_HPP
#define MPR_MPRDISPATCHER_HPP

#include <data-units/Precision.hpp>
#include <utilities/MPRErrorHandler.hpp>


using namespace mpr::precision;

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
                MPR_API_EXCEPTION("C++ Error : Type Undefined Dispatcher",     \
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
               MPR_API_EXCEPTION("C++ Error : Type Undefined Dispatcher",      \
                                (int)PRECISION);                               \
               }                                                               \
          }  ;                                                                 \

/** Instantiators for Template functions with a given return type
 * (One template argument)
 **/
#define SIMPLE_INSTANTIATE(RETURNTYPE, __FUN__, ...) \
        template RETURNTYPE __FUN__<float16> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template RETURNTYPE __FUN__<float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template RETURNTYPE __FUN__<double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \

/** Instantiators for Template functions with return type the same as dispatching
 *  type (One template argument)
 **/
#define SIMPLE_INSTANTIATE_WITH_RETURN(__FUN__, ...) \
        template float16 __FUN__<float16> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \
        template float __FUN__<float> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ;\
        template double __FUN__<double> (FIRST(__VA_ARGS__)REST(__VA_ARGS__)) ; \

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


#endif //MPR_MPRDISPATCHER_HPP
