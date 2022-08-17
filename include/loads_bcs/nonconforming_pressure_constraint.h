#ifndef MPM_NONCONFORMING_PRESSURE_CONSTRAINT_H_
#define MPM_NONCONFORMING_PRESSURE_CONSTRAINT_H_

#include "function_base.h"

namespace mpm {

//! NonconformingPressureConstraint class to store non-conforming pressure
//! constraint \brief NonconformingPressureConstraint class to store a
//! constraint \details NonconformingPressureConstraint stores the constraint as
//! a static value
class NonconformingPressureConstraint {
 public:
  // Constructor
  //! \param[in] bounding_box Bounding box [xmin, xmax, ymin, ymax, zmin, zmax]
  //! \param[in] inside True if surface is inside bounding box
  //! \param[in] mfunction Math function
  //! \param[in] pressure Pressure
  //! \param[in] traction 1 or 0 for active or inactive directions
  //! \param[in] traction_grad Gradient of traction values
  NonconformingPressureConstraint(
      const std::vector<double> bounding_box, const bool inside,
      const std::shared_ptr<mpm::FunctionBase>& mfunction,
      const double pressure, const std::vector<double> traction,
      const std::vector<double> traction_grad)
      : bounding_box_{bounding_box},
        inside_{inside},
        pressure_fn_{mfunction},
        pressure_{pressure},
        traction_{traction},
        traction_grad_{traction_grad} {};

  // Bounding box around surface
  std::vector<double> bounding_box() const { return bounding_box_; }
  // True if surface is inside bounding box
  bool inside() const { return inside_; }
  // Pressure
  double pressure(double current_time) const {
    return pressure_ * (this->pressure_fn_)->value(current_time);
  }
  // Traction
  std::vector<double> traction() const { return traction_; }
  // Traction gradient
  std::vector<double> traction_grad() const { return traction_grad_; }

 private:
  // Bounding box around surface
  std::vector<double> bounding_box_;
  // True if surface is inside bounding box
  bool inside_;
  // Pressure magnitude
  double pressure_;
  // Math function
  std::shared_ptr<mpm::FunctionBase> pressure_fn_;
  // Traction
  std::vector<double> traction_;
  // Traction gradient
  std::vector<double> traction_grad_;
};
}  // namespace mpm
#endif  // MPM_NONCONFORMING_PRESSURE_CONSTRAINT_H_
