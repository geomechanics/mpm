#ifndef MPM_TEMPERATURE_CONSTRAINT_H_
#define MPM_TEMPERATURE_CONSTRAINT_H_

namespace mpm {

//! TemperatureConstraint class to store temperature constraint on a set
//! \brief TemperatureConstraint class to store a constraint on a set
//! \details TemperatureConstraint stores the constraint as a static value
class TemperatureConstraint {
 public:
  // Constructor
  //! \param[in] setid  set id
  //! \param[in] temperature Constraint temperature
  TemperatureConstraint(int setid, unsigned phase, double temperature)
      : setid_{setid}, phase_{phase}, temperature_{temperature} {};

  // Set id
  int setid() const { return setid_; }

  // Return phase
  unsigned phase() const { return phase_; }

  // Return temperature
  double temperature() const { return temperature_; }

 private:
  // ID
  int setid_;
  // Phase
  unsigned phase_;
  // Temperature
  double temperature_;
};  // namespace mpm
}  // namespace mpm
#endif  // MPM_TEMPERATURE_CONSTRAINT_H_
