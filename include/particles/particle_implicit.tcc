//! Map particle mass, momentum and inertia to nodes
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_mass_momentum_inertia_to_nodes() noexcept {
  // Map mass and momentum to nodes
  this->map_mass_momentum_to_nodes();

  // Map inertia to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_inertia(true, mpm::ParticlePhase::Solid,
                              mass_ * shapefn_[i] * acceleration_);
  }
}

//! Function to reinitialise material to be run at the beginning of each time
template <unsigned Tdim>
void mpm::Particle<Tdim>::initialise_constitutive_law() noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);

  // Reset material to be Elastic
  material_[mpm::ParticlePhase::Solid]->initialise(
      &state_variables_[mpm::ParticlePhase::Solid]);

  // Compute initial consititutive matrix
  this->constitutive_matrix_ =
      material_[mpm::ParticlePhase::Solid]->compute_consistent_tangent_matrix(
          stress_, previous_stress_, dstrain_, this,
          &state_variables_[mpm::ParticlePhase::Solid]);
}

//! Map inertial force
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_inertial_force() noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);

  // Compute nodal inertial forces
  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodes_[i]->update_external_force(
        true, mpm::ParticlePhase::Solid,
        (-1. * nodes_[i]->acceleration(mpm::ParticlePhase::Solid) * mass_ *
         shapefn_(i)));
}

//! Map mass and material stiffness matrix to cell (used in poisson equation
//! LHS)
template <unsigned Tdim>
inline bool mpm::Particle<Tdim>::map_stiffness_matrix_to_cell(
    double newmark_beta, double dt, bool quasi_static) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Compute material stiffness matrix
    this->map_material_stiffness_matrix_to_cell();

    // Compute mass matrix
    if (!quasi_static) this->map_mass_matrix_to_cell(newmark_beta, dt);

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map material stiffness matrix to cell (used in equilibrium equation LHS)
template <unsigned Tdim>
inline bool mpm::Particle<Tdim>::map_material_stiffness_matrix_to_cell() {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Reduce constitutive relations matrix depending on the dimension
    const Eigen::MatrixXd reduced_dmatrix =
        this->reduce_dmatrix(constitutive_matrix_);

    // Calculate B matrix
    const Eigen::MatrixXd bmatrix = this->compute_bmatrix();

    // Compute local material stiffness matrix
    cell_->compute_local_material_stiffness_matrix(bmatrix, reduced_dmatrix,
                                                   volume_);
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Compute B matrix
template <>
inline Eigen::MatrixXd mpm::Particle<1>::compute_bmatrix() noexcept {
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(1, this->nodes_.size());
  bmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, i) = dn_dx_(i, 0);
  }
  return bmatrix;
}

// Compute B matrix
template <>
inline Eigen::MatrixXd mpm::Particle<2>::compute_bmatrix() noexcept {
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(3, 2 * this->nodes_.size());
  bmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, 2 * i) = dn_dx_(i, 0);
    bmatrix(2, 2 * i) = dn_dx_(i, 1);
    bmatrix(1, 2 * i + 1) = dn_dx_(i, 1);
    bmatrix(2, 2 * i + 1) = dn_dx_(i, 0);
  }
  return bmatrix;
}

// Compute B matrix
template <>
inline Eigen::MatrixXd mpm::Particle<3>::compute_bmatrix() noexcept {
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(6, 3 * this->nodes_.size());
  bmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, 3 * i) = dn_dx_(i, 0);
    bmatrix(3, 3 * i) = dn_dx_(i, 1);
    bmatrix(5, 3 * i) = dn_dx_(i, 2);

    bmatrix(1, 3 * i + 1) = dn_dx_(i, 1);
    bmatrix(3, 3 * i + 1) = dn_dx_(i, 0);
    bmatrix(4, 3 * i + 1) = dn_dx_(i, 2);

    bmatrix(2, 3 * i + 2) = dn_dx_(i, 2);
    bmatrix(4, 3 * i + 2) = dn_dx_(i, 1);
    bmatrix(5, 3 * i + 2) = dn_dx_(i, 0);
  }
  return bmatrix;
}

//! Reduce constitutive relations matrix depending on the dimension
template <>
inline Eigen::MatrixXd mpm::Particle<1>::reduce_dmatrix(
    const Eigen::MatrixXd& dmatrix) noexcept {

  // Convert to 1x1 matrix in 1D
  Eigen::MatrixXd dmatrix1x1;
  dmatrix1x1.resize(1, 1);
  dmatrix1x1(0, 0) = dmatrix(0, 0);

  return dmatrix1x1;
}

//! Reduce constitutive relations matrix depending on the dimension
template <>
inline Eigen::MatrixXd mpm::Particle<2>::reduce_dmatrix(
    const Eigen::MatrixXd& dmatrix) noexcept {

  // Convert to 3x3 matrix in 2D
  Eigen::MatrixXd dmatrix3x3;
  dmatrix3x3.resize(3, 3);
  dmatrix3x3(0, 0) = dmatrix(0, 0);
  dmatrix3x3(0, 1) = dmatrix(0, 1);
  dmatrix3x3(0, 2) = dmatrix(0, 3);
  dmatrix3x3(1, 0) = dmatrix(1, 0);
  dmatrix3x3(1, 1) = dmatrix(1, 1);
  dmatrix3x3(1, 2) = dmatrix(1, 3);
  dmatrix3x3(2, 0) = dmatrix(3, 0);
  dmatrix3x3(2, 1) = dmatrix(3, 1);
  dmatrix3x3(2, 2) = dmatrix(3, 3);

  return dmatrix3x3;
}

//! Reduce constitutive relations matrix depending on the dimension
template <>
inline Eigen::MatrixXd mpm::Particle<3>::reduce_dmatrix(
    const Eigen::MatrixXd& dmatrix) noexcept {
  return dmatrix;
}

//! Map mass matrix to cell (used in poisson equation LHS)
template <unsigned Tdim>
inline bool mpm::Particle<Tdim>::map_mass_matrix_to_cell(double newmark_beta,
                                                         double dt) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Compute local mass matrix
    cell_->compute_local_mass_matrix(shapefn_, volume_,
                                     mass_density_ / (newmark_beta * dt * dt));
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::Particle<1>::compute_strain_increment(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rincrement
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 1, 1> displacement = nodes_[i]->displacement(phase);
    strain_increment[0] += dn_dx(i, 0) * displacement[0];
  }

  if (std::fabs(strain_increment(0)) < 1.E-15) strain_increment[0] = 0.;
  return strain_increment;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::Particle<2>::compute_strain_increment(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain increment
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 2, 1> displacement = nodes_[i]->displacement(phase);
    strain_increment[0] += dn_dx(i, 0) * displacement[0];
    strain_increment[1] += dn_dx(i, 1) * displacement[1];
    strain_increment[3] +=
        dn_dx(i, 1) * displacement[0] + dn_dx(i, 0) * displacement[1];
  }

  if (std::fabs(strain_increment[0]) < 1.E-15) strain_increment[0] = 0.;
  if (std::fabs(strain_increment[1]) < 1.E-15) strain_increment[1] = 0.;
  if (std::fabs(strain_increment[3]) < 1.E-15) strain_increment[3] = 0.;
  return strain_increment;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::Particle<3>::compute_strain_increment(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain increment
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 3, 1> displacement = nodes_[i]->displacement(phase);
    strain_increment[0] += dn_dx(i, 0) * displacement[0];
    strain_increment[1] += dn_dx(i, 1) * displacement[1];
    strain_increment[2] += dn_dx(i, 2) * displacement[2];
    strain_increment[3] +=
        dn_dx(i, 1) * displacement[0] + dn_dx(i, 0) * displacement[1];
    strain_increment[4] +=
        dn_dx(i, 2) * displacement[1] + dn_dx(i, 1) * displacement[2];
    strain_increment[5] +=
        dn_dx(i, 2) * displacement[0] + dn_dx(i, 0) * displacement[2];
  }

  for (unsigned i = 0; i < strain_increment.size(); ++i)
    if (std::fabs(strain_increment[i]) < 1.E-15) strain_increment[i] = 0.;
  return strain_increment;
}

// Compute strain and volume of the particle using nodal displacement
template <unsigned Tdim>
void mpm::Particle<Tdim>::compute_strain_volume_newmark() noexcept {
  // Compute the volume at the previous time step
  this->volume_ /= (1. + dvolumetric_strain_);
  this->mass_density_ *= (1. + dvolumetric_strain_);

  // Compute strain increment from previous time step
  this->dstrain_ =
      this->compute_strain_increment(dn_dx_, mpm::ParticlePhase::Solid);

  // Updated volumetric strain increment
  this->dvolumetric_strain_ = this->dstrain_.head(Tdim).sum();

  // Update volume using volumetric strain increment
  this->volume_ *= (1. + dvolumetric_strain_);
  this->mass_density_ /= (1. + dvolumetric_strain_);
}

// Compute stress using implicit updating scheme
template <unsigned Tdim>
void mpm::Particle<Tdim>::compute_stress_newmark() noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);
  // Clone state variables
  auto temp_state_variables = state_variables_[mpm::ParticlePhase::Solid];
  // Calculate stress
  this->stress_ = (this->material())
                      ->compute_stress(previous_stress_, dstrain_, this,
                                       &temp_state_variables);

  // Compute current consititutive matrix
  this->constitutive_matrix_ =
      material_[mpm::ParticlePhase::Solid]->compute_consistent_tangent_matrix(
          stress_, previous_stress_, dstrain_, this, &temp_state_variables);
}

// Compute updated position of the particle by Newmark scheme
template <unsigned Tdim>
void mpm::Particle<Tdim>::compute_updated_position_newmark(double dt) noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);
  // Get interpolated nodal displacement and acceleration
  Eigen::Matrix<double, Tdim, 1> nodal_displacement =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  Eigen::Matrix<double, Tdim, 1> nodal_acceleration =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodal_displacement.noalias() +=
        shapefn_[i] * nodes_[i]->displacement(mpm::ParticlePhase::Solid);
    nodal_acceleration.noalias() +=
        shapefn_[i] * nodes_[i]->acceleration(mpm::ParticlePhase::Solid);
  }

  // Update particle velocity from interpolated nodal acceleration
  this->velocity_.noalias() +=
      0.5 * (this->acceleration_ + nodal_acceleration) * dt;

  // Update acceleration
  this->acceleration_ = nodal_acceleration;

  // New position  current position + displacement increment
  this->coordinates_.noalias() += nodal_displacement;
  // Update displacement
  this->displacement_.noalias() += nodal_displacement;
}

// Update stress and strain after convergence of Newton-Raphson iteration
template <unsigned Tdim>
void mpm::Particle<Tdim>::update_stress_strain() noexcept {
  // Update converged stress
  this->stress_ =
      (this->material())
          ->compute_stress(previous_stress_, dstrain_, this,
                           &state_variables_[mpm::ParticlePhase::Solid]);

  // Update initial stress of the time step
  this->previous_stress_ = this->stress_;

  // Update total strain
  this->strain_.noalias() += this->dstrain_;

  // Reset strain increment
  this->dstrain_.setZero();
  this->dvolumetric_strain_ = 0.;
}

// Assign acceleration to the particle
template <unsigned Tdim>
bool mpm::Particle<Tdim>::assign_acceleration(
    const Eigen::Matrix<double, Tdim, 1>& acceleration) {
  // Assign acceleration
  acceleration_ = acceleration;
  return true;
}

// Compute deformation gradient increment of the particle
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::Particle<1>::compute_deformation_gradient_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& displacement = nodes_[i]->displacement(phase);
    deformation_gradient_increment(0, 0) += dn_dx(i, 0) * displacement[0];
  }

  if (std::fabs(deformation_gradient_increment(0, 0) - 1.) < 1.E-15)
    deformation_gradient_increment(0, 0) = 1.;
  return deformation_gradient_increment;
}

// Compute deformation gradient increment of the particle
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::Particle<2>::compute_deformation_gradient_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& displacement = nodes_[i]->displacement(phase);
    deformation_gradient_increment.block(0, 0, 2, 2).noalias() +=
        displacement * dn_dx.row(i);
  }

  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      if (i != j && std::fabs(deformation_gradient_increment(i, j)) < 1.E-15)
        deformation_gradient_increment(i, j) = 0.;
      if (i == j &&
          std::fabs(deformation_gradient_increment(i, j) - 1.) < 1.E-15)
        deformation_gradient_increment(i, j) = 1.;
    }
  }
  return deformation_gradient_increment;
}

// Compute deformation gradient increment of the particle
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::Particle<3>::compute_deformation_gradient_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& displacement = nodes_[i]->displacement(phase);
    deformation_gradient_increment.noalias() += displacement * dn_dx.row(i);
  }

  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      if (i != j && std::fabs(deformation_gradient_increment(i, j)) < 1.E-15)
        deformation_gradient_increment(i, j) = 0.;
      if (i == j &&
          std::fabs(deformation_gradient_increment(i, j) - 1.) < 1.E-15)
        deformation_gradient_increment(i, j) = 1.;
    }
  }
  return deformation_gradient_increment;
}
