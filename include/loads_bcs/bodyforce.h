#ifndef MPM_BODYFORCE_H_
#define MPM_BODYFORCE_H_

//! Alias for JSON
#include "json.hpp"
using Json = nlohmann::json;

#include "function_base.h"

namespace mpm {

//! BodyForce class to store the body force on a set
//! \brief BodyForce class to store the body force on a set
//! \details BodyForce stores the body force on a set using mathematical
//! functions, so the force can vary dynamically with time
class BodyForce {
 public:
  // Constructor
  //! \param[setid] setid  set id
  //! \param[in] mfunction Math function if defined
  //! \param[dir] dir Direction of body force
  //! \param[force] force  body force magnitude
  BodyForce(int setid, const std::shared_ptr<mpm::FunctionBase>& force_fn,
            unsigned dir, double force)
      : setid_{setid}, force_fn_{force_fn}, dir_{dir}, force_{force} {};

  // Set id
  int setid() const { return setid_; }

  // Direction
  unsigned dir() const { return dir_; }

  // Return body force
  double amplitude(double current_time) const {
    // Static force when no math function is defined
    double scalar = (this->force_fn_ != nullptr)
                        ? (this->force_fn_)->value(current_time)
                        : 1.0;
    return force_ * scalar;
  }

 private:
  // ID
  int setid_;
  // Math function
  std::shared_ptr<mpm::FunctionBase> force_fn_;
  // Direction
  unsigned dir_;
  // Body force
  double force_;
};

}  // namespace mpm

#endif  // MPM_BODYFORCE_H_