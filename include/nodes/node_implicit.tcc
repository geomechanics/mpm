//! Initialise implicit nodal properties
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::initialise_implicit() noexcept {
  this->initialise();
  // Specific variables for implicit solver
  inertia_.setZero();
  displacement_.setZero();
}

//! Initialise nodal force during Newton-Raphson iteration
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::initialise_force() noexcept {
  // nodal forces
  external_force_.setZero();
  internal_force_.setZero();
}

//! Assign nodal inertia
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_inertia(
    bool update, unsigned phase,
    const Eigen::Matrix<double, Tdim, 1>& inertia) noexcept {
  // Assert
  assert(phase < Tnphases);

  // Decide to update or assign
  const double factor = (update == true) ? 1. : 0.;

  // Update/assign inertia
  node_mutex_.lock();
  inertia_.col(phase) = inertia_.col(phase) * factor + inertia;
  node_mutex_.unlock();
}

//! Compute velocity and acceleration from momentum and inertia
//! velocity = momentum / mass
//! acceleration = inertia / mass
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::compute_velocity_acceleration() {
  const double tolerance = 1.E-16;
  for (unsigned phase = 0; phase < Tnphases; ++phase) {
    if (mass_(phase) > tolerance) {
      velocity_.col(phase) = momentum_.col(phase) / mass_(phase);
      acceleration_.col(phase) = inertia_.col(phase) / mass_(phase);

      // Check to see if value is below threshold
      for (unsigned i = 0; i < velocity_.rows(); ++i)
        if (std::abs(velocity_.col(phase)(i)) < 1.E-15)
          velocity_.col(phase)(i) = 0.;

      for (unsigned i = 0; i < acceleration_.rows(); ++i)
        if (std::abs(acceleration_.col(phase)(i)) < 1.E-15)
          acceleration_.col(phase)(i) = 0.;
    }
  }

  // Apply displacement constraints, which also sets velocity and acceleration
  // to 0, as well as momentum and inertia.
  this->apply_displacement_constraints();
}

//! Update velocity and acceleration by Newmark scheme
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_velocity_acceleration_newmark(
    unsigned phase, double newmark_beta, double newmark_gamma, double dt) {
  const double tolerance = 1.E-16;
  //! Compute velocity and acceleration at the previous time step
  VectorDim previous_velocity;
  VectorDim previous_acceleration;
  previous_velocity.setZero();
  previous_acceleration.setZero();
  if (mass_(phase) > tolerance) {
    previous_velocity = momentum_.col(phase) / mass_(phase);
    previous_acceleration = inertia_.col(phase) / mass_(phase);
  }

  //! Update of velocity and acceleration
  velocity_.col(phase) =
      newmark_gamma / newmark_beta / dt * displacement_.col(phase) -
      (newmark_gamma / newmark_beta - 1.) * previous_velocity -
      0.5 * dt * (newmark_gamma / newmark_beta - 2.) * previous_acceleration;

  acceleration_.col(phase) =
      1. / newmark_beta / dt / dt * displacement_.col(phase) -
      1. / newmark_beta / dt * previous_velocity -
      (0.5 / newmark_beta - 1.) * previous_acceleration;

  // Check to see if value is below threshold
  for (unsigned i = 0; i < velocity_.rows(); ++i)
    if (std::abs(velocity_.col(phase)(i)) < 1.E-15)
      velocity_.col(phase)(i) = 0.;

  for (unsigned i = 0; i < acceleration_.rows(); ++i)
    if (std::abs(acceleration_.col(phase)(i)) < 1.E-15)
      acceleration_.col(phase)(i) = 0.;
}

//! Update displacement increment at the node
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_displacement_increment(
    const Eigen::VectorXd& displacement_increment, unsigned phase,
    const unsigned nactive_node) {

  for (unsigned i = 0; i < Tdim; ++i) {
    displacement_.col(phase)(i) +=
        displacement_increment(nactive_node * i + active_id_);
  }
}

//! Assign displacement constraints
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
bool mpm::Node<Tdim, Tdof, Tnphases>::assign_displacement_constraint(
    const unsigned dir, const double displacement,
    const std::shared_ptr<FunctionBase>& function) {
  bool status = true;
  try {
    //! Constrain directions can take values between 0 and Dim * Nphases
    if (dir < (Tdim * Tnphases))
      this->displacement_constraints_.insert(std::make_pair<unsigned, double>(
          static_cast<unsigned>(dir), static_cast<double>(displacement)));
    else
      throw std::runtime_error("Constraint direction is out of bounds");

    // Assign displacement function
    if (function != nullptr)
      this->displacement_function_.insert(
          std::make_pair<unsigned, std::shared_ptr<FunctionBase>>(
              static_cast<unsigned>(dir),
              static_cast<std::shared_ptr<FunctionBase>>(function)));

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Apply displacement constraints
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::apply_displacement_constraints() {
  // Set displacement constraint
  for (const auto& constraint : this->displacement_constraints_) {
    // Direction value in the constraint (0, Dim * Nphases)
    const unsigned dir = constraint.first;
    // Direction: dir % Tdim (modulus)
    const auto direction = static_cast<unsigned>(dir % Tdim);
    // Phase: Integer value of division (dir / Tdim)
    const auto phase = static_cast<unsigned>(dir / Tdim);

    if (generic_boundary_constraints_)
      throw std::runtime_error(
          "Generic boundary constraints is not available for displacement "
          "constraints!");

    // Displacement constraints are applied on Cartesian boundaries
    this->displacement_(direction, phase) = constraint.second;
    // Set velocity and acceleration to 0 in direction of displacement
    // constraint
    this->velocity_(direction, phase) = 0.;
    this->acceleration_(direction, phase) = 0.;

    // Set momentum and inertia to 0 in direction of displacement constraint
    this->momentum_(direction, phase) = 0.;
    this->inertia_(direction, phase) = 0.;
  }
}