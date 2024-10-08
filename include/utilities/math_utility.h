#ifndef MPM_MATH_UTILITY_H_
#define MPM_MATH_UTILITY_H_

#include <cmath>

#include "data_types.h"

namespace mpm {
namespace math {
//! Convert 2nd-order symmetric tensor from voigt notation to full matrix form
//! with template dimension
//! \param[in] voigt_tensor Tensor in Voigt notation
//! \param[in] shear_multiplier Shear multiplier during transformation: 1.0
//! (for stress), 0.5 (for strain)
template <unsigned Tdim>
inline const Eigen::Matrix<double, Tdim, Tdim> matrix_form(
    const Eigen::Matrix<double, 6, 1>& voigt_tensor, double shear_multiplier);

//! Convert 2nd-order symmetric tensor from voigt notation to full matrix form
//! \param[in] voigt_tensor Tensor in Voigt notation
//! \param[in] shear_multiplier Shear multiplier during transformation: 1.0
//! (for stress), 0.5 (for strain)
inline const Eigen::Matrix<double, 3, 3> matrix_form(
    const Eigen::Matrix<double, 6, 1>& voigt_tensor, double shear_multiplier);

//! Convert 2nd-order symmetric tensor from full matrix form to voigt notation
//! with template dimension
//! \param[in] matrix_tensor Tensor in matrix form
//! \param[in] shear_multiplier Shear multiplier during transformation: 1.0
//! (for stress), 2.0 (for strain)
template <unsigned Tdim>
inline const Eigen::Matrix<double, 6, 1> voigt_form(
    const Eigen::Matrix<double, Tdim, Tdim>& matrix_tensor,
    double shear_multiplier);

//! Convert 2nd-order symmetric tensor from full matrix form to voigt notation
//! \param[in] matrix_tensor Tensor in matrix form
//! \param[in] shear_multiplier Shear multiplier during transformation: 1.0
//! (for stress), 2.0 (for strain)
inline const Eigen::Matrix<double, 6, 1> voigt_form(
    const Eigen::Matrix<double, 3, 3>& matrix_tensor, double shear_multiplier);

//! Compute principal stress/strain from given stress/strain in matrix form
//! \param[in] matrix_tensor Tensor in matrix form
//! \param[in] directors Eigen vector describing the rotation of the tensor to
//! principal coordinates
//! \retval Sorted principal stresses from high to low
inline const Eigen::Matrix<double, 3, 1> principal_tensor(
    const Eigen::Matrix<double, 3, 3>& matrix_tensor);

//! Compute principal stress/strain from given stress/strain in matrix form
//! \param[in] matrix_tensor Tensor in matrix form
//! \retval Sorted principal stresses from high to low
inline const Eigen::Matrix<double, 3, 1> principal_tensor(
    const Eigen::Matrix<double, 3, 3>& matrix_tensor,
    Eigen::Matrix<double, 3, 3>& directors);

}  // namespace math
}  // namespace mpm

#include "math_utility.tcc"

#endif  // MPM_MATH_UTILITY_H_
