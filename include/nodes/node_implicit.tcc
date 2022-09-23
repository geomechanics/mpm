//! Initialise implicit nodal properties
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::initialise_implicit() noexcept {
  this->initialise();
  // Specific variables for implicit solver
  inertia_.setZero();
  displacement_.setZero();
  predictor_displacement_.setZero();
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

//! Update displacement by explicit Newmark scheme
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::compute_predictor_displacement_newmark(
    unsigned phase, double newmark_beta, double newmark_gamma, double dt) {
  const double tolerance = 1.E-16;
  //! Compute velocity and acceleration at the previous time step
  VectorDim previous_velocity = VectorDim::Zero();
  VectorDim previous_acceleration = VectorDim::Zero();
  if (mass_(phase) > tolerance) {
    previous_velocity = momentum_.col(phase) / mass_(phase);
    previous_acceleration = inertia_.col(phase) / mass_(phase);
  }

  // Compute new acceleration explicitly
  VectorDim explicit_acceleration =
      (this->external_force_.col(phase) + this->internal_force_.col(phase)) /
      this->mass_(phase);

  // Compute new displacement using Newmark
  predictor_displacement_.col(phase).noalias() +=
      dt * previous_velocity +
      dt * dt / 2.0 *
          ((1 - 2.0 * newmark_beta) * previous_acceleration +
           2.0 * newmark_beta * explicit_acceleration);

  try {
    // Loop over displacement constraint
    for (const auto& constraint : this->displacement_constraints_) {
      // Direction value in the constraint (0, Dim * Nphases)
      const unsigned dir = constraint.first;
      // Direction: dir % Tdim (modulus)
      const auto direction = static_cast<unsigned>(dir % Tdim);
      // Phase: Integer value of division (dir / Tdim)
      const auto cphase = static_cast<unsigned>(dir / Tdim);
      // Exit loop if phase is not the same
      if (cphase != phase) continue;

      if (!generic_boundary_constraints_) {
        // Velocity constraints are applied on Cartesian boundaries
        predictor_displacement_.col(phase)(direction) = constraint.second;
      } else {
        throw std::runtime_error(
            "Rotational function to impose displacement boundary is not yet "
            "available!");
      }
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }

  // Check to see if value is below threshold
  for (unsigned i = 0; i < predictor_displacement_.rows(); ++i)
    if (std::abs(predictor_displacement_.col(phase)(i)) < 1.E-15)
      predictor_displacement_.col(phase)(i) = 0.;
}