/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/

#ifndef MPCR_FLOATINGPOINTHANDLER_HPP
#define MPCR_FLOATINGPOINTHANDLER_HPP


#ifdef USE_CUDA
#include <cuda_fp16.h>
typedef half float16;
#define USING_HALF 1

#else
typedef int float16;
#endif




#endif //MPCR_FLOATINGPOINTHANDLER_HPP
