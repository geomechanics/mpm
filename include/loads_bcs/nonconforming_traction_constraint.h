#ifndef MPM_NONCONFORMING_TRACTION_CONSTRAINT_H_
#define MPM_NONCONFORMING_TRACTION_CONSTRAINT_H_

#include "function_base.h"

namespace mpm {

//! NonconformingTractionConstraint class to store non-conforming traction
//! constraint
//! \brief NonconformingTractionConstraint class to store a constraint
//! \details NonconformingTractionConstraint stores the constraint as a static
//! value
template <unsigned Tdim>
class NonconformingTractionConstraint {
 public:
  // Constructor
  //! \param[in] bounding_box Bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
  //! \param[in] datum Datum for hydrostatic free surface
  //! \param[in] fluid_density Fluid density for hydrostatic
  //! \param[in] gravity Gravity for hydrostatic
  //! \param[in] hydrostatic True if hydrostatic pressure, false for constant
  //! \param[in] inside True if surface is inside bounding box
  //! \param[in] mfunction Math function
  //! \param[in] pressure Traction if constant
  NonconformingTractionConstraint(
      const std::vector<double> bounding_box, const double datum,
      const double fluid_density, const double gravity, const bool hydrostatic,
      const bool inside, const std::shared_ptr<mpm::FunctionBase>& mfunction,
      const double pressure, const unsigned number_gaussian_points)
      : bounding_box_{bounding_box},
        datum_{datum},
        fluid_density_{fluid_density},
        gravity_{gravity},
        hydrostatic_{hydrostatic},
        inside_{inside},
        pressure_fn_{mfunction},
        pressure_{pressure},
        number_gaussian_points_{number_gaussian_points} {
    if (Tdim == 2) {
      switch (number_gaussian_points_) {
        case 1:
          quadrature_ =
              Factory<mpm::Quadrature<Tdim>>::instance()->create("QQUAD1");
          break;
        case 2:
          quadrature_ =
              Factory<mpm::Quadrature<Tdim>>::instance()->create("QQUAD2");
          break;
        case 3:
          quadrature_ =
              Factory<mpm::Quadrature<Tdim>>::instance()->create("QQUAD3");
          break;
        case 4:
          quadrature_ =
              Factory<mpm::Quadrature<Tdim>>::instance()->create("QQUAD4");
          break;
        default:
          quadrature_ =
              Factory<mpm::Quadrature<Tdim>>::instance()->create("QQUAD1");
          break;
      }
    } else if (Tdim == 3) {
      switch (number_gaussian_points_) {
        case 1:
          quadrature_ =
              Factory<mpm::Quadrature<Tdim>>::instance()->create("QHEX1");
          break;
        case 2:
          quadrature_ =
              Factory<mpm::Quadrature<Tdim>>::instance()->create("QHEX2");
          break;
        case 3:
          quadrature_ =
              Factory<mpm::Quadrature<Tdim>>::instance()->create("QHEX3");
          break;
        case 4:
          quadrature_ =
              Factory<mpm::Quadrature<Tdim>>::instance()->create("QHEX4");
          break;
        default:
          quadrature_ =
              Factory<mpm::Quadrature<Tdim>>::instance()->create("QHEX1");
          break;
      }
    }
  };

  // Bounding box around surface
  std::vector<double> bounding_box() const { return bounding_box_; }
  // Datum for hydrostatic free surface
  double datum() const { return datum_; }
  // Fluid density for hydrostatic free surface
  double fluid_density() const { return fluid_density_; }
  // Gravity for hydrostatic free surface
  double gravity() const { return gravity_; }
  // True if pressure from hydrostatic, false if constant pressure
  bool hydrostatic() const { return hydrostatic_; }
  // True if surface of interest is inside bounding box
  bool inside() const { return inside_; }
  // Pressure
  double pressure(double current_time) const {
    // Max pressure when no math function is defined
    double scalar = (this->pressure_fn_ != nullptr)
                        ? (this->pressure_fn_)->value(current_time)
                        : 1.0;
    return (-1. * pressure_ * scalar);
  }
  // Number of Gaussian points
  unsigned number_gaussian_points() const { return number_gaussian_points_; }

  //! Return quadrature points
  Eigen::MatrixXd quadratures() const { return quadrature_->quadratures(); };
  //! Return weights
  Eigen::VectorXd weights() const { return quadrature_->weights(); };

 private:
  // Bounding box around surface
  std::vector<double> bounding_box_;
  // Datum for hydrostatic free surface
  double datum_;
  // Fluid density for hydrostatic free surface
  double fluid_density_;
  // Gravity for hydrostatic free surface
  double gravity_;
  // True if pressure from hydrostatic,
  bool hydrostatic_;
  // True if surface is inside bounding box
  bool inside_;
  // Pressure magnitude
  double pressure_;
  // Number of Gaussian points
  unsigned number_gaussian_points_{1};
  // Quadrature
  std::shared_ptr<Quadrature<Tdim>> quadrature_{nullptr};
  // Math function
  std::shared_ptr<mpm::FunctionBase> pressure_fn_;
};
}  // namespace mpm
#endif  // MPM_NONCONFORMING_TRACTION_CONSTRAINT_H_
