//! Map particle mass, momentum and inertia to nodes
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_mass_momentum_inertia_to_nodes(
    mpm::VelocityUpdate velocity_update) noexcept {
  // Map mass and momentum to nodes
  this->map_mass_momentum_to_nodes(velocity_update);

  switch (velocity_update) {
    case mpm::VelocityUpdate::APIC:
      this->map_inertia_to_nodes_affine();
      break;
    case mpm::VelocityUpdate::ASFLIP:
      this->map_inertia_to_nodes_affine();
      break;
    case mpm::VelocityUpdate::TPIC:
      this->map_inertia_to_nodes_taylor();
      break;
    default:
      // Map inertia to nodes
      for (unsigned i = 0; i < nodes_.size(); ++i) {
        nodes_[i]->update_inertia(true, this->phase(),
                                  mass_ * shapefn_[i] * acceleration_);
      }
      break;
  }
}

//! Map particle inertia to nodes for affine transformation
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_inertia_to_nodes_affine() noexcept {

  // Initialise Mapping matrix if necessary
  if (mapping_matrix_.rows() != Tdim) {
    mapping_matrix_.resize(Tdim, Tdim);
    mapping_matrix_.setZero();
  }

  // Shape tensor computation for APIC
  Eigen::MatrixXd shape_tensor;
  shape_tensor.resize(Tdim, Tdim);
  shape_tensor.setZero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    const auto& branch_vector = nodes_[i]->coordinates() - this->coordinates_;
    shape_tensor.noalias() +=
        shapefn_[i] * branch_vector * branch_vector.transpose();
  }

  // Map inertia to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Initialise map acceleration
    VectorDim map_acceleration = acceleration_;
    map_acceleration.noalias() +=
        mapping_matrix_ * shape_tensor.inverse() *
        (nodes_[i]->coordinates() - this->coordinates_);

    // Map inertia
    nodes_[i]->update_inertia(true, this->phase(),
                              mass_ * shapefn_[i] * map_acceleration);
  }
}

//! Map particle inertia to nodes for approximate taylor expansion
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_inertia_to_nodes_taylor() noexcept {

  // Initialise Mapping matrix if necessary
  if (mapping_matrix_.rows() != Tdim) {
    mapping_matrix_.resize(Tdim, Tdim);
    mapping_matrix_.setZero();
  }

  // Map mass and momentum to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Initialise map acceleration
    VectorDim map_acceleration = acceleration_;
    map_acceleration.noalias() +=
        mapping_matrix_ * (nodes_[i]->coordinates() - this->coordinates_);

    // Map inertia
    nodes_[i]->update_inertia(true, this->phase(),
                              mass_ * shapefn_[i] * map_acceleration);
  }
}

//! Function to reinitialise material to be run at the beginning of each time
template <unsigned Tdim>
void mpm::Particle<Tdim>::initialise_constitutive_law(double dt) noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);

  // Reset material to be Elastic
  material_[mpm::ParticlePhase::Solid]->initialise(
      &state_variables_[mpm::ParticlePhase::Solid]);

  // Compute initial consititutive matrix
  this->constitutive_matrix_ =
      material_[mpm::ParticlePhase::Solid]->compute_consistent_tangent_matrix(
          stress_, previous_stress_, dstrain_, this,
          &state_variables_[mpm::ParticlePhase::Solid], dt);
}

//! Map inertial force
template <unsigned Tdim>
void mpm::Particle<Tdim>::map_inertial_force(double bossak_alpha) noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);

  // Compute nodal inertial forces
  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodes_[i]->update_external_force(
        true, this->phase(),
        (-1. * nodes_[i]->acceleration(this->phase()) * mass_ * shapefn_(i)));
}

//! Map mass and material stiffness matrix to cell (used in Newton-raphson
//! equation LHS)
template <unsigned Tdim>
inline bool mpm::Particle<Tdim>::map_stiffness_matrix_to_cell(
    double newmark_beta, double newmark_gamma, double bossak_alpha, double dt,
    bool quasi_static) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Compute material stiffness matrix
    this->map_material_stiffness_matrix_to_cell(dt);

    // Compute mass matrix
    if (!quasi_static)
      this->map_mass_matrix_to_cell(newmark_beta, bossak_alpha, dt);

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map material stiffness matrix to cell (used in equilibrium equation LHS)
template <unsigned Tdim>
inline bool mpm::Particle<Tdim>::map_material_stiffness_matrix_to_cell(
    double dt) {
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
                                                         double bossak_alpha,
                                                         double dt) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Compute local mass matrix
    cell_->compute_local_mass_matrix(
        shapefn_, volume_,
        (1 - bossak_alpha) * mass_density_ / (newmark_beta * dt * dt));
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::Particle<1>::compute_strain_increment(
    const Eigen::MatrixXd& dn_dx, unsigned phase, double dt) noexcept {
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
    const Eigen::MatrixXd& dn_dx, unsigned phase, double dt) noexcept {
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
    const Eigen::MatrixXd& dn_dx, unsigned phase, double dt) noexcept {
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
void mpm::Particle<Tdim>::compute_strain_volume_newmark(double dt) noexcept {
  // Compute the volume at the previous time step
  this->volume_ /= (1. + dvolumetric_strain_);
  this->mass_density_ *= (1. + dvolumetric_strain_);

  // Compute deformation gradient increment from previous time step
  this->deformation_gradient_increment_ =
      this->compute_deformation_gradient_increment(this->dn_dx_,
                                                   mpm::ParticlePhase::Solid);

  // Compute strain increment from previous time step
  this->dstrain_ =
      this->compute_strain_increment(dn_dx_, mpm::ParticlePhase::Solid, dt);

  // Updated volumetric strain increment
  this->dvolumetric_strain_ = this->dstrain_.head(Tdim).sum();

  // Update volume using volumetric strain increment
  this->volume_ *= (1. + dvolumetric_strain_);
  this->mass_density_ /= (1. + dvolumetric_strain_);
}

// Compute stress using implicit updating scheme
template <unsigned Tdim>
void mpm::Particle<Tdim>::compute_stress_newmark(double dt) noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);
  // Clone state variables
  auto temp_state_variables = state_variables_[mpm::ParticlePhase::Solid];
  // Calculate stress
  this->stress_ = (this->material())
                      ->compute_stress(previous_stress_, dstrain_, this,
                                       &temp_state_variables, dt);

  // Compute current consititutive matrix
  this->constitutive_matrix_ =
      material_[mpm::ParticlePhase::Solid]->compute_consistent_tangent_matrix(
          stress_, previous_stress_, dstrain_, this, &temp_state_variables, dt);
}

// Compute updated position of the particle by Newmark scheme
template <unsigned Tdim>
void mpm::Particle<Tdim>::compute_updated_position_newmark(
    double dt, double newmark_gamma, unsigned step,
    mpm::VelocityUpdate velocity_update, double blending_ratio) noexcept {
  switch (velocity_update) {
    case mpm::VelocityUpdate::FLIP:
      this->compute_updated_position_newmark_flip(dt, newmark_gamma, step,
                                                  blending_ratio);
      break;
    case mpm::VelocityUpdate::PIC:
      this->compute_updated_position_newmark_pic();
      break;
  }
}

// Compute updated position of the particle by Newmark-FLIP scheme
template <unsigned Tdim>
void mpm::Particle<Tdim>::compute_updated_position_newmark_flip(
    double dt, double newmark_gamma, unsigned step,
    double blending_ratio) noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);
  // Get interpolated nodal displacement, velocity, and acceleration
  Eigen::Matrix<double, Tdim, 1> nodal_displacement =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  Eigen::Matrix<double, Tdim, 1> nodal_velocity =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  Eigen::Matrix<double, Tdim, 1> nodal_acceleration =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodal_displacement.noalias() +=
        shapefn_[i] * nodes_[i]->displacement(this->phase());
    nodal_velocity.noalias() +=
        shapefn_[i] * nodes_[i]->velocity(this->phase());
    nodal_acceleration.noalias() +=
        shapefn_[i] * nodes_[i]->acceleration(this->phase());
  }

  // Update particle velocity from interpolated nodal acceleration
  if (step > 0)
    this->velocity_.noalias() += ((1.0 - newmark_gamma) * this->acceleration_ +
                                  newmark_gamma * nodal_acceleration) *
                                 dt;
  else
    this->velocity_.noalias() += nodal_acceleration * dt;

  // If intermediate scheme is considered
  this->velocity_ = blending_ratio * this->velocity_ +
                    (1.0 - blending_ratio) * nodal_velocity;

  // Update acceleration
  this->acceleration_ = nodal_acceleration;

  // New position  current position + displacement increment
  this->coordinates_.noalias() += nodal_displacement;
  // Update displacement
  this->displacement_.noalias() += nodal_displacement;
}

// Compute updated position of the particle by Newmark-PIC scheme
template <unsigned Tdim>
void mpm::Particle<Tdim>::compute_updated_position_newmark_pic() noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);
  // Get interpolated nodal displacement, velocity, and acceleration
  Eigen::Matrix<double, Tdim, 1> nodal_displacement =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  Eigen::Matrix<double, Tdim, 1> nodal_velocity =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  Eigen::Matrix<double, Tdim, 1> nodal_acceleration =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodal_displacement.noalias() +=
        shapefn_[i] * nodes_[i]->displacement(this->phase());
    nodal_velocity.noalias() +=
        shapefn_[i] * nodes_[i]->velocity(this->phase());
    nodal_acceleration.noalias() +=
        shapefn_[i] * nodes_[i]->acceleration(this->phase());
  }

  // Update velocity and acceleration
  this->velocity_ = nodal_velocity;
  this->acceleration_ = nodal_acceleration;

  // New position  current position + displacement increment
  this->coordinates_.noalias() += nodal_displacement;
  // Update displacement
  this->displacement_.noalias() += nodal_displacement;
}

// Update stress and strain after convergence of Newton-Raphson iteration
template <unsigned Tdim>
void mpm::Particle<Tdim>::update_stress_strain(double dt) noexcept {
  // Update converged stress
  this->stress_ =
      (this->material())
          ->compute_stress(previous_stress_, dstrain_, this,
                           &state_variables_[mpm::ParticlePhase::Solid], dt);

  // Update initial stress of the time step
  this->previous_stress_ = this->stress_;

  // Update total strain
  this->strain_.noalias() += this->dstrain_;

  // Reset strain increment
  this->dstrain_.setZero();
  this->dvolumetric_strain_ = 0.;

  // Update deformation gradient
  this->deformation_gradient_ =
      this->deformation_gradient_increment_ * this->deformation_gradient_;

  // Reset deformation gradient increment
  this->deformation_gradient_increment_.setIdentity();
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

//! Compute ASFLIP beta parameter using displacement
template <unsigned Tdim>
inline double mpm::Particle<Tdim>::compute_asflip_beta() noexcept {
  double beta = 1.0;
  // Check if particle is located nearby imposed boundary
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& disp_constraints = nodes_[i]->displacement_constraints();
    if (disp_constraints.size() > 0) {
      beta = 0.0;
      break;
    }
  }

  // Check if the incremental Jacobian is in compressive mode
  const auto def_grad_increment =
      this->compute_deformation_gradient_increment(this->dn_dx_, this->phase());
  const double J = def_grad_increment.determinant();
  if (J < 1.0) beta = 0.0;

  return beta;
}