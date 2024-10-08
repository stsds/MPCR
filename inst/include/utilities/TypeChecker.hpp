/**
 * Copyright (c) 2023, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 * MPCR is an R package provided by the STSDS group at KAUST
 *
 **/
#include <complex>

/**
 * @brief
 * Type trait function to check if it is a complex type.
 *
 * @tparam T
 * Type to test
 */
template<typename T>
struct is_complex_t : public std::false_type {
};

/**
 * @brief
 * Type trait function to check if it is a complex type.
 *
 * @tparam T
 * Type to test
 */
template<typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {
};

/**
 * @brief
 * Type trait function to check if it is a complex type.
 *
 * @tparam T
 * Type to test
 *
 * @return
 * True if it is a complex type.
 */
template<typename T>
constexpr bool is_complex() {
    return is_complex_t<T>::value;
}

/**
 * @brief
 * Type trait function to check if it is a complex type of single precision.
 *
 * @tparam T
 * Type to test
 */
template<typename T>
struct is_complex_float_t : public std::false_type {
};

/**
 * @brief
 * Type trait function to check if it is a complex type of single precision.
 *
 * @tparam T
 * Type to test
 */
template<>
struct is_complex_float_t<std::complex<float>> : public std::true_type {
};

/**
 * @brief
 * Type trait function to check if it is a complex type of single precision.
 *
 * @tparam T
 * Type to test
 *
 * @return
 * True if it is a complex type of single precision.
 */
template<typename T>
constexpr bool is_complex_float() {
    return is_complex_float_t<T>::value;
}

/**
 * @brief
 * Type trait function to check if it is a double type.
 *
 * @tparam T
 * Type to test
 */
template<typename T>
struct is_double_t : public std::false_type {
};

/**
 * @brief
 * Type trait function to check if it is a double type.
 *
 * @tparam T
 * Type to test
 */
template<>
struct is_double_t<double> : public std::true_type {
};

/**
 * @brief
 * Type trait function to check if it is double precision.
 *
 * @tparam T
 * Type to test
 *
 * @return
 * True if it is a double precision.
 */
template<typename T>
constexpr bool is_double() {
    return is_double_t<T>::value;
}

/**
 * @brief
 * Type trait function to check if it is a single type.
 *
 * @tparam T
 * Type to test
 */
template<typename T>
struct is_float_t : public std::false_type {
};

/**
 * @brief
 * Type trait function to check if it is a single type.
 *
 * @tparam T
 * Type to test
 */
template<>
struct is_float_t<float> : public std::true_type {
};


/**
 * @brief
 * Type trait function to check if it is single precision.
 *
 * @tparam T
 * Type to test
 *
 * @return
 * True if it is a double precision.
 */
template<typename T>
constexpr bool is_float() {
    return is_float_t<T>::value;
}


/**
 * @brief
 * Type trait function to check if it is a half type.
 *
 * @tparam T
 * Type to test
 */
template<typename T>
struct is_half_t : public std::false_type {
};

/**
 * @brief
 * Type trait function to check if it is a half type.
 *
 * @tparam T
 * Type to test
 */
#ifdef USING_HALF
template<>
struct is_half_t<float16> : public std::true_type {
};
#else
template<>
struct is_half_t<float16> : public std::false_type {
};
#endif

/**
 * @brief
 * Type trait function to check if it is half precision.
 *
 * @tparam T
 * Type to test
 *
 * @return
 * True if it is a double precision.
 */
template<typename T>
constexpr bool is_half() {
    return is_half_t<T>::value;
}