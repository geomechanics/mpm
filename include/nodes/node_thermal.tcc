
//! Initialise nodal properties
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::initialise_thermal() noexcept {
  temperature_.setZero();
  temperature_rate_.setZero();
  temperature_ddot_.setZero();
  temperature_increment_.setZero();
  heat_capacity_.setZero();
  heat_.setZero();
  heat_rate_.setZero();
  heat_ddot_.setZero();
  internal_heat_.setZero();
  external_heat_.setZero();
}

//! Assign/update heat capacity at nodes from particle
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_heat_capacity(
    bool update, unsigned phase, double heat_capacity) noexcept {
  // Assert
  assert(phase < Tnphases);
  
  // Decide to update or assign
  const double factor = (update == true) ? 1. : 0.;
  
  // Update/assign mass
  node_mutex_.lock();
  heat_capacity_(phase) = heat_capacity_(phase) * factor + heat_capacity;
  node_mutex_.unlock();
}

//! Assign/update heat at nodes from particle
//  Heat = \rho * c * T
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_heat(
    bool update, unsigned phase, double heat) noexcept {
  // Assert
  assert(phase < Tnphases);

  // Decide to update or assign
  const double factor = (update == true) ? 1. : 0.;

  // Update/assign heat
  node_mutex_.lock();
  heat_(phase) = heat_(phase) * factor + heat;
  node_mutex_.unlock();  
}

//! Assign/update internal heat
//  Internal_heat = heat_rate + heat conduction + heat_convection...
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_internal_heat(
    bool update, unsigned phase, const double internal_heat) noexcept {
  // Assert
  assert(phase < Tnphases);

  // Decide to update or assign
  const double factor = (update == true) ? 1. : 0.;

  // Update/assign internal heat
  node_mutex_.lock();
  internal_heat_(phase) = internal_heat_(phase) * factor + internal_heat;  
  node_mutex_.unlock();  
}

//! Assign/update external heat
//  External heat = plastic work + external heat source + boundary heat flux...
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_external_heat(
    bool update, unsigned phase, const double external_heat) noexcept {
  // Assert
  assert(phase < Tnphases);

  // Decide to update or assign
  const double factor = (update == true) ? 1. : 0.;

  // Update/assign external heat
  node_mutex_.lock();
  external_heat_(phase) = external_heat_(phase) * factor + external_heat;
  node_mutex_.unlock();  
}

//! Compute nodal temperature from nodal heat and heat capacity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::compute_temperature_explicit( 
                              unsigned phase, double dt, Index step) noexcept {
  const double tolerance = 1.E-15;
  if (this->heat_capacity_(phase) > tolerance) {
    this->temperature_(phase) = 
                this->heat_(phase) / this->heat_capacity_(phase);
  }
  // Apply temperature boundary conditions
  // temperature = boundary temeprature, 
  this->apply_temperature_constraint(phase, dt, step);
}

//! Compute nodal temperature rate and nodal temperature
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
bool mpm::Node<Tdim, Tdof, Tnphases>::update_temperature_explicit(
    unsigned phase, double dt, Index step) noexcept {
  bool status = false;
  const double tolerance = 1.0E-15;

  if (this->heat_capacity_(phase) > tolerance) {
    // temperature dot = (total_heat / heat_capacity)
    this->temperature_rate_(phase) =
            (this->internal_heat_(phase) + this->external_heat_(phase)) / 
            this->heat_capacity_(phase);
    this->temperature_ += this->temperature_rate_ * dt;
    status = true;
  }
  this->apply_temperature_constraint(phase, dt, step); 
  return status;
}

//! Assign temperature constraint
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
bool mpm::Node<Tdim, Tdof, Tnphases>::assign_temperature_constraint(
    unsigned phase, double temperature,
    const std::shared_ptr<FunctionBase>& function) {
  bool status = true;
  try {
    // Constrain directions can take values between 0 and Tnphases
    if (phase < Tnphases) {
      this->temperature_constraints_.insert(std::make_pair<unsigned, double>(
          static_cast<unsigned>(phase), static_cast<double>(temperature)));
      // Assign temperature function
      if (function != nullptr)
        this->temperature_function_.insert(
            std::make_pair<unsigned, std::shared_ptr<FunctionBase>>(
                static_cast<unsigned>(phase),
                static_cast<std::shared_ptr<FunctionBase>>(function)));
    } else
      throw std::runtime_error("Temperature constraint phase is invalid");

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Apply temperature constraint
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::apply_temperature_constraint(
    unsigned phase, double dt, Index step) noexcept {
  // Assert
  assert(phase < Tnphases);

  if (temperature_constraints_.find(phase) != temperature_constraints_.end()) {
    const double scalar =
        (temperature_function_.find(phase) != temperature_function_.end())
            ? temperature_function_[phase]->value(step * dt)
            : 1.0;
    this->temperature_(phase) = scalar * temperature_constraints_[phase];
    this->temperature_rate_(phase) = 0;
    this->temperature_ddot_(phase) = 0;
    this->temperature_increment_(phase) = 0;
  }
}