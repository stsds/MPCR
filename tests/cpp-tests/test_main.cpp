/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MMPR is an R package provided by the STSDS group at KAUST
 *
 **/
#define CATCH_CONFIG_RUNNER

#include <libraries/catch/catch.hpp>


void runCatchTests(int argc, char *const argv[]) {
    Catch::Session().run(argc, argv);
}


int main(int argc, char *const argv[]) {
    runCatchTests(argc, argv);
    return 0;
}
