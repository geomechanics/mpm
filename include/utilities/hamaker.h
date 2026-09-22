#ifndef MPM_HAMAKER_H_
#define MPM_HAMAKER_H_

#include <algorithm>
#include <cmath>
#include <limits>

namespace mpm {
namespace materials {

// Mean surface-to-surface gap between grains from the packing fraction, picture in notes
//! \tparam Tdim Dimension (exponent of the lattice scaling is 1/Tdim) (still unsure if this works properly)
//! \param[in] phi Current packing fraction, solved for in dpm.tcc
//! \param[in] phi_ref Packing fraction at which grains touch (gap = 0), need to determine whether to use critical_packing_fraction or dialtion stuff
//! \param[in] grain_diameter Mean grain diameter
//! \retval gap Gap (>= 0)
template <unsigned Tdim>
inline double hamaker_gap(double phi, double phi_ref, double grain_diameter) {
  // Guard against empty or negative packing fraction
  if (phi <= std::numeric_limits<double>::epsilon())
    return std::numeric_limits<double>::max();
  const double gap =
      grain_diameter * (std::pow(phi_ref / phi, 1.0 / Tdim) - 1.0);
  return std::max(gap, 0.0);
}

//! Hamaker (van der Waals) attractive force between two grains
//! \param[in] gap Surface-to-surface gap
//! \param[in] grain_diameter Mean grain diameter
//! \param[in] hamaker_const Hamaker constant A, will compute via paper defintion in dpm.tcc
//! \param[in] delta_min Minimum separation (contact cutoff, must be > 0) HOW TO DO THIS
//! \param[in] delta_max Maximum gap beyond which the force is zero, HOW DETERMINE THESE
//! \retval force Attractive force (>= 0)
inline double hamaker_force(double gap, double grain_diameter,
                            double hamaker_const, double delta_min,
                            double delta_max) {
  if (gap > delta_max) return 0.0;
  // Regularised separation: finite force at contact
  const double s = std::max(gap, 0.0) + delta_min;
  return hamaker_const * grain_diameter / (12.0 * s * s);
}

}  // namespace materials
}  // namespace mpm

#endif  // MPM_HAMAKER_H_
