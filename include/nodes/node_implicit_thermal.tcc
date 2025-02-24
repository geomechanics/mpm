//! Initialise nodal properties
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::initialise_heat() noexcept {
  internal_heat_.setZero();
  external_heat_.setZero();  
}

//! Assign/update first time derivitive of heat
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_heat_rate(
    bool update, unsigned phase, double heat_rate) noexcept {
  // Assert
  assert(phase < Tnphases);

  // Decide to update or assign
  const double factor = (update == true) ? 1. : 0.;

  // Update/assign heat
  node_mutex_.lock();
  heat_rate_(phase) = heat_rate_(phase) * factor + heat_rate;
  node_mutex_.unlock();  
}

//! Assign/update second time derivitive of heat
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_heat_ddot(
    bool update, unsigned phase, double heat_ddot) noexcept {
  // Assert
  assert(phase < Tnphases);

  // Decide to update or assign
  const double factor = (update == true) ? 1. : 0.;

  // Update/assign heat
  node_mutex_.lock();
  heat_ddot_(phase) = heat_ddot_(phase) * factor + heat_ddot;
  node_mutex_.unlock();  
}

//! Compute nodal temperature from heat and heat capacity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::compute_temperature_implicit(
                            unsigned phase, double dt, Index step) noexcept {
  const double tolerance = 1.E-15;
  if (this->heat_capacity_(phase) > tolerance) {
    this->temperature_(phase) = this->heat_(phase) / 
                                this->heat_capacity_(phase);
    this->temperature_rate_(phase) = this->heat_rate_(phase) / 
                                    this->heat_capacity_(phase);
    
    // Check to see if value is below threshold
    if (std::abs(temperature_rate_(phase)) < 1.E-15)
      temperature_rate_(phase) = 0.;
  }
  // Apply temperature boundary conditions
  // temperature = boundary temeprature, 
  // temperature_rate = 0 
  this->apply_temperature_constraint(phase, dt, step);
}

//! Update temperature variables by Newmark scheme
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_temperature_variables_newmark(
        unsigned phase, double newmark_beta, double newmark_gamma, 
        double dt, double step) {
  const double tolerance = 1.E-16;
  //! Compute temeprature rate at the previous time step
  double previous_temperature_rate;
  double previous_temperature_ddot;
  previous_temperature_rate = 0;
  previous_temperature_ddot = 0;

  if (mass_(phase) > tolerance) {
    previous_temperature_rate = heat_rate_(phase) / heat_capacity_(phase);
    previous_temperature_ddot = heat_ddot_(phase) / heat_capacity_(phase);
  }

  //! Update of temeprature rate -- Backward Euler method
  // temperature_rate_(phase) = 1 / dt * temperature_increment_(phase);

  //! Update of temeprature rate -- Newmark-bata method
  temperature_rate_(phase) =
      newmark_gamma / newmark_beta / dt * temperature_increment_(phase) -
      (newmark_gamma / newmark_beta - 1.) * previous_temperature_rate -
      0.5 * dt * (newmark_gamma / newmark_beta - 2.) * 
      previous_temperature_ddot;

  temperature_ddot_(phase) =
      1. / newmark_beta / dt / dt * temperature_increment_(phase) -
      1. / newmark_beta / dt * previous_temperature_rate -
      (0.5 / newmark_beta - 1.) * previous_temperature_ddot;

  // Apply temperature boundary conditions
  // temperature_rate = 0
  // temperature_ddot = 0
  // this->apply_temperature_constraint(phase, dt, step);

  // Check to see if value is below threshold
  if (std::abs(temperature_rate_(phase)) < 1.E-15)
    temperature_rate_(phase) = 0.;

  if (std::abs(temperature_ddot_(phase)) < 1.E-15)
    temperature_ddot_(phase) = 0.;
}

//! Update temperature increment at the node
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_temperature_increment(
    const Eigen::VectorXd& temperature_increment, unsigned phase,
    double dt, Index step) {

  temperature_increment_(phase) += temperature_increment(active_id_);
  temperature_(phase) += temperature_increment(active_id_);

  this->apply_temperature_constraint(phase, dt, step);  
}