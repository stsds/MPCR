/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

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