#ifndef MPM_VELOCITY_CONSTRAINT_H_
#define MPM_VELOCITY_CONSTRAINT_H_

namespace mpm {

//! VelocityConstraint class to store velocity constraint on a set
//! \brief VelocityConstraint class to store a constraint on a set
//! \details VelocityConstraint stores the constraint as a static value
class VelocityConstraint {
 public:
  // Constructor
  //! \param[in] setid  set id
  //! \param[in] velocity_fn Math function if defined
  //! \param[in] dir Direction of constraint load
  //! \param[in] velocity Constraint  velocity
  VelocityConstraint(int setid,
                     const std::shared_ptr<mpm::FunctionBase>& velocity_fn,
                     unsigned dir, double velocity)
      : setid_{setid},
        velocity_fn_{velocity_fn},
        dir_{dir},
        velocity_{velocity} {};

  // Set id
  int setid() const { return setid_; }

  // Direction
  unsigned dir() const { return dir_; }

  // Return velocity
  double velocity(double current_time) const {
    // Static load when no math function is defined
    double scalar = (this->velocity_fn_ != nullptr)
                        ? (this->velocity_fn_)->value(current_time)
                        : 1.0;
    return velocity_ * scalar;
  }

 private:
  // ID
  int setid_;
  // Math function
  std::shared_ptr<mpm::FunctionBase> velocity_fn_;
  // Direction
  unsigned dir_;
  // Velocity
  double velocity_;
};
}  // namespace mpm
#endif  // MPM_VELOCITY_CONSTRAINT_H_
