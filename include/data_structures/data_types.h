#ifndef MPM_DATA_TYPES_H_
#define MPM_DATA_TYPES_H_

#ifdef USE_MKL
#define EIGEN_USE_MKL_ALL
#endif

#include <Eigen/Dense>

namespace mpm {

//! Global index type for the node
using Index = unsigned long long;

//! Return zero
template <typename Ttype>
Ttype zero();

//! Zero
template <>
inline Eigen::Matrix<double, 2, 1> zero() {
  return Eigen::Matrix<double, 2, 1>::Zero();
}

//! Zero
template <>
inline Eigen::Matrix<double, 3, 1> zero() {
  return Eigen::Matrix<double, 3, 1>::Zero();
}

//! Zero
template <>
inline double zero() {
  return 0.;
}

//! Position type
//! None: No position is specified
//! Corner: Nodes at boundary corners
//! Edge: Nodes along boundary edges
//! Face: Nodes on boundary faces
enum class Position { None, Corner, Edge, Face };

//! Stress rate type
//! None: Cauchy rate (non-objective)
//! Jaumann: Jaumann rate of Cauchy stress tensor (objective)
enum class StressRate { None, Jaumann };

//! Velocity update type
//! Flip: fluid-implicit-particle (acceleration update)
//! Pic: particle-in-cell (velocity update)
//! Asflip: Affine Separable flip
//! Tpic: Taylor pic
//! Apic: Affine pic
enum class VelocityUpdate { FLIP, PIC, ASFLIP, TPIC, APIC };

}  // namespace mpm

#endif  // MPM_DATA_TYPES_H_
