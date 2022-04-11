#ifndef MPM_TETRAHEDRON_QUADRATURE_H_
#define MPM_TETRAHEDRON_QUADRATURE_H_

#include <exception>

#include <Eigen/Dense>

#include "quadrature.h"

//! MPM namespace
namespace mpm {

//! Tetrahedron quadrature class derived from Quadrature class
//! \brief Quadrature (gauss points) for a tetrahedron element
//! source: http://solidmechanics.org/Text/Chapter8_1/Chapter8_1.php
//! \tparam Tdim Dimension
//! \tparam Tnquadratures number of quadratures
template <unsigned Tdim, unsigned Tnquadratures>
class TetrahedronQuadrature : public Quadrature<Tdim> {

 public:
  TetrahedronQuadrature() : mpm::Quadrature<Tdim>() {
    static_assert(Tdim == 3, "Invalid dimension for a 3D tetrahedron element");
    static_assert((Tnquadratures == 1) || (Tnquadratures == 4),
                  "Invalid number of quadratures");
  }

  //! Return quadrature points
  //! \param[out] qpoints Quadrature points in local coordinates
  Eigen::MatrixXd quadratures() const override;

  //! Return weights
  //! \param[out] weights Weights for quadrature points
  Eigen::VectorXd weights() const override;
};

}  // namespace mpm

#include "tetrahedron_quadrature.tcc"

#endif  // MPM_TETRAHEDRON_QUADRATURE_H_
