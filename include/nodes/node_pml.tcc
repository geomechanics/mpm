//! Compute PML velocity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::compute_pml_velocity() {
  const double tolerance = 1.E-16;

  // Damped mass vector
  VectorDim damped_mass =
      property_handle_->property("damped_masses", prop_id_, 0, Tdim);

  for (unsigned phase = 0; phase < Tnphases; ++phase) {
    if (mass_(phase) > tolerance) {
      for (unsigned i = 0; i < Tdim; i++)
        velocity_.col(phase)(i) = momentum_.col(phase)(i) / damped_mass(i);

      // Check to see if value is below threshold
      for (unsigned i = 0; i < velocity_.rows(); ++i)
        if (std::abs(velocity_.col(phase)(i)) < 1.E-15)
          velocity_.col(phase)(i) = 0.;
    }
  }

  // Apply velocity constraints, which also sets acceleration to 0,
  // when velocity is set.
  this->apply_velocity_constraints();

  // Apply pml displacement constraints.
  this->apply_pml_displacement_constraints();
}

//! Compute PML velocity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::compute_pml_velocity_acceleration() {
  const double tolerance = 1.E-16;

  // Damped mass vector
  VectorDim damped_mass =
      property_handle_->property("damped_masses", prop_id_, 0, Tdim);

  for (unsigned phase = 0; phase < Tnphases; ++phase) {
    if (mass_(phase) > tolerance) {
      for (unsigned i = 0; i < Tdim; i++) {
        velocity_.col(phase)(i) = momentum_.col(phase)(i) / damped_mass(i);
        acceleration_.col(phase)(i) = inertia_.col(phase)(i) / damped_mass(i);
      }

      // Check to see if value is below threshold
      for (unsigned i = 0; i < velocity_.rows(); ++i)
        if (std::abs(velocity_.col(phase)(i)) < 1.E-15)
          velocity_.col(phase)(i) = 0.;
      for (unsigned i = 0; i < acceleration_.rows(); ++i)
        if (std::abs(acceleration_.col(phase)(i)) < 1.E-15)
          acceleration_.col(phase)(i) = 0.;
    }
  }

  // Apply pml displacement constraints.
  this->apply_pml_displacement_constraints();
}

//! Compute PML nodal acceleration and velocity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
bool mpm::Node<Tdim, Tdof, Tnphases>::compute_pml_acceleration_velocity(
    unsigned phase, double dt, double damping_factor) noexcept {
  bool status = false;
  const double tolerance = 1.0E-15;

  // Damped mass vector
  VectorDim damped_mass =
      property_handle_->property("damped_masses", prop_id_, 0, Tdim);

  if (mass_(phase) > tolerance) {
    // acceleration = (unbalaced force / mass)
    for (unsigned i = 0; i < Tdim; i++)
      this->acceleration_.col(phase)(i) =
          (this->external_force_.col(phase)(i) +
           this->internal_force_.col(phase)(i)) /
              damped_mass(i) -
          this->velocity_.col(phase)(i) * damping_factor;
    // Apply acceleration constraints
    this->apply_acceleration_constraints();

    // Velocity += acceleration * dt
    this->velocity_.col(phase).noalias() += this->acceleration_.col(phase) * dt;
    // Apply velocity constraints, which also sets acceleration to 0,
    // when velocity is set.
    this->apply_velocity_constraints();

    // Set a threshold
    for (unsigned i = 0; i < Tdim; ++i)
      if (std::abs(velocity_.col(phase)(i)) < tolerance)
        velocity_.col(phase)(i) = 0.;
    for (unsigned i = 0; i < Tdim; ++i)
      if (std::abs(acceleration_.col(phase)(i)) < tolerance)
        acceleration_.col(phase)(i) = 0.;
    status = true;
  }
  return status;
}

//! Apply pml displacement constraints
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::apply_pml_displacement_constraints() {
  // Set displacement constraint
  for (const auto& constraint : this->displacement_constraints_) {
    // Direction value in the constraint (0, Dim * Nphases)
    const unsigned dir = constraint.first;
    // Direction: dir % Tdim (modulus)
    const auto direction = static_cast<unsigned>(dir % Tdim);

    if (generic_boundary_constraints_)
      throw std::runtime_error(
          "Generic boundary constraints is not available for displacement "
          "constraints!");

    // Displacement constraints are applied on Cartesian boundaries
    VectorDim damped_mass_disp = property_handle_->property(
        "damped_mass_displacements", prop_id_, 0, Tdim);
    damped_mass_disp(direction) = constraint.second;
    property_handle_->assign_property("damped_mass_displacements", prop_id_, 0,
                                      damped_mass_disp, Tdim);

    // Displacement constraints of historical boundary
    // j = 1
    VectorDim damped_mass_disp_j1 = property_handle_->property(
        "damped_mass_displacements_j1", prop_id_, 0, Tdim);
    damped_mass_disp_j1(direction) = 0.0;
    property_handle_->assign_property("damped_mass_displacements_j1", prop_id_,
                                      0, damped_mass_disp_j1, Tdim);

    // j = 2
    VectorDim damped_mass_disp_j2 = property_handle_->property(
        "damped_mass_displacements_j2", prop_id_, 0, Tdim);
    damped_mass_disp_j2(direction) = 0.0;
    property_handle_->assign_property("damped_mass_displacements_j2", prop_id_,
                                      0, damped_mass_disp_j2, Tdim);

    // j = 3
    VectorDim damped_mass_disp_j3 = property_handle_->property(
        "damped_mass_displacements_j3", prop_id_, 0, Tdim);
    damped_mass_disp_j3(direction) = 0.0;
    property_handle_->assign_property("damped_mass_displacements_j3", prop_id_,
                                      0, damped_mass_disp_j3, Tdim);

    // j = 4
    VectorDim damped_mass_disp_j4 = property_handle_->property(
        "damped_mass_displacements_j4", prop_id_, 0, Tdim);
    damped_mass_disp_j4(direction) = 0.0;
    property_handle_->assign_property("damped_mass_displacements_j4", prop_id_,
                                      0, damped_mass_disp_j4, Tdim);
  }
}

//! Update velocity and acceleration by Newmark scheme if pml is used
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::Node<Tdim, Tdof, Tnphases>::update_velocity_acceleration_newmark_pml(
    unsigned phase, double newmark_beta, double newmark_gamma, double dt) {
  const double tolerance = 1.E-16;
  //! Compute velocity and acceleration at the previous time step
  VectorDim previous_velocity = VectorDim::Zero();
  VectorDim previous_acceleration = VectorDim::Zero();

  // Damped mass vector
  VectorDim damped_mass =
      property_handle_->property("damped_masses", prop_id_, 0, Tdim);

  if (mass_(phase) > tolerance) {
    for (unsigned i = 0; i < Tdim; i++) {
      previous_velocity(i) = momentum_.col(phase)(i) / damped_mass(i);
      previous_acceleration(i) = inertia_.col(phase)(i) / damped_mass(i);
    }
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

//! Return previous displacement of pml nodes
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
Eigen::Matrix<double, Tdim, 1>
    mpm::Node<Tdim, Tdof, Tnphases>::previous_pml_displacement(
        unsigned t_index) const {
  const double tolerance = 1.E-16;
  VectorDim damped_mass =
      property_handle_->property("damped_masses", prop_id_, 0, Tdim);

  // Time index switch
  VectorDim damped_mass_disp;
  switch (t_index) {
    case 0:  // Displacement at tn
      damped_mass_disp = property_handle_->property("damped_mass_displacements",
                                                    prop_id_, 0, Tdim);
      break;
    case 1:  // Displacement at tn - dt
      damped_mass_disp = property_handle_->property(
          "damped_mass_displacements_j1", prop_id_, 0, Tdim);
      break;
    case 2:  // Displacement at tn - 2*dt
      damped_mass_disp = property_handle_->property(
          "damped_mass_displacements_j2", prop_id_, 0, Tdim);
      break;
    case 3:  // Displacement at tn - 2*dt
      damped_mass_disp = property_handle_->property(
          "damped_mass_displacements_j3", prop_id_, 0, Tdim);
      break;
    case 4:  // Displacement at tn - 2*dt
      damped_mass_disp = property_handle_->property(
          "damped_mass_displacements_j4", prop_id_, 0, Tdim);
      break;
    default:
      throw std::runtime_error("Invalid time index for pml displacement");
      break;
  }

  VectorDim previous_pml_displacement = VectorDim::Zero();
  if (mass_(mpm::NodePhase::NSinglePhase) > tolerance) {
    for (unsigned i = 0; i < Tdim; i++) {
      previous_pml_displacement(i) = damped_mass_disp(i) / damped_mass(i);
    }
  }
  return previous_pml_displacement;
};

//! Return previous velocity of pml nodes
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
Eigen::Matrix<double, Tdim, 1>
    mpm::Node<Tdim, Tdof, Tnphases>::previous_pml_velocity() const {
  const double tolerance = 1.E-16;
  VectorDim damped_mass =
      property_handle_->property("damped_masses", prop_id_, 0, Tdim);
  VectorDim previous_pml_velocity = VectorDim::Zero();

  if (mass_(mpm::NodePhase::NSinglePhase) > tolerance) {
    for (unsigned i = 0; i < Tdim; i++) {
      previous_pml_velocity(i) =
          momentum_.col(mpm::NodePhase::NSinglePhase)(i) / damped_mass(i);
    }
  }
  return previous_pml_velocity;
};

//! Return previous acceleration of pml nodes
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
Eigen::Matrix<double, Tdim, 1>
    mpm::Node<Tdim, Tdof, Tnphases>::previous_pml_acceleration() const {
  const double tolerance = 1.E-16;
  VectorDim damped_mass =
      property_handle_->property("damped_masses", prop_id_, 0, Tdim);
  VectorDim previous_pml_acceleration = VectorDim::Zero();

  if (mass_(mpm::NodePhase::NSinglePhase) > tolerance) {
    for (unsigned i = 0; i < Tdim; i++) {
      previous_pml_acceleration(i) =
          inertia_.col(mpm::NodePhase::NSinglePhase)(i) / damped_mass(i);
    }
  }
  return previous_pml_acceleration;
};