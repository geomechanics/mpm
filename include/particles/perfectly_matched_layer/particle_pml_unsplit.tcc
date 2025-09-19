//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleUPML<Tdim>::ParticleUPML(Index id, const VectorDim& coord)
    : mpm::ParticlePML<Tdim>(id, coord) {
  // Logger
  std::string logger =
      "particle_upml" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticleUPML<Tdim>::ParticleUPML(Index id, const VectorDim& coord,
                                      bool status)
    : mpm::ParticlePML<Tdim>(id, coord, status) {
  //! Logger
  std::string logger =
      "particle_upml" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Map particle mass and momentum to nodes
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_mass_momentum_to_nodes(
    mpm::VelocityUpdate velocity_update) noexcept {

  switch (velocity_update) {
    case mpm::VelocityUpdate::APIC:
      this->map_mass_momentum_to_nodes_affine();
      break;
    case mpm::VelocityUpdate::ASFLIP:
      this->map_mass_momentum_to_nodes_affine();
      break;
    case mpm::VelocityUpdate::TPIC:
      this->map_mass_momentum_to_nodes_taylor();
      break;
    default:
      // Check if particle mass is set
      assert(mass_ != std::numeric_limits<double>::max());

      // Damping Functions
      const auto& Fe = this->evanescent_damping_functions();

      // Damping scalar
      double f_M = Fe.prod();

      // Map mass and momentum to nodes
      for (unsigned i = 0; i < nodes_.size(); ++i) {
        // Map mass and momentum
        nodes_[i]->update_mass(true, mpm::ParticlePhase::SinglePhase,
                               mass_ * shapefn_[i]);
        nodes_[i]->update_momentum(true, mpm::ParticlePhase::SinglePhase,
                                   mass_ * shapefn_[i] * velocity_ * f_M);
      }
      break;
  }
}

//! Map particle mass and momentum to nodes for affine transformation
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_mass_momentum_to_nodes_affine() noexcept {
  // Check if particle mass is set
  assert(mass_ != std::numeric_limits<double>::max());

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

  // Damping Functions
  const auto& Fe = this->evanescent_damping_functions();

  // Damping scalar
  double f_M = Fe.prod();

  // Map mass and momentum to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Initialise map velocity
    VectorDim map_velocity = velocity_;
    map_velocity.noalias() += mapping_matrix_ * shape_tensor.inverse() *
                              (nodes_[i]->coordinates() - this->coordinates_);

    // Map mass and momentum
    nodes_[i]->update_mass(true, mpm::ParticlePhase::Solid,
                           mass_ * shapefn_[i]);
    nodes_[i]->update_momentum(true, mpm::ParticlePhase::Solid,
                               mass_ * shapefn_[i] * map_velocity * f_M);
  }
}

//! Map particle mass and momentum to nodes for approximate taylor expansion
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_mass_momentum_to_nodes_taylor() noexcept {
  // Check if particle mass is set
  assert(mass_ != std::numeric_limits<double>::max());

  // Initialise Mapping matrix if necessary
  if (mapping_matrix_.rows() != Tdim) {
    mapping_matrix_.resize(Tdim, Tdim);
    mapping_matrix_.setZero();
  }

  // Damping Functions
  const auto& Fe = this->evanescent_damping_functions();

  // Damping scalar
  double f_M = Fe.prod();

  // Map mass and momentum to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Initialise map velocity
    VectorDim map_velocity = velocity_;
    map_velocity.noalias() +=
        mapping_matrix_ * (nodes_[i]->coordinates() - this->coordinates_);

    // Map mass and momentum
    nodes_[i]->update_mass(true, mpm::ParticlePhase::Solid,
                           mass_ * shapefn_[i]);
    nodes_[i]->update_momentum(true, mpm::ParticlePhase::Solid,
                               mass_ * shapefn_[i] * map_velocity * f_M);
  }
}

//! Map particle mass, momentum and inertia to nodes
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_mass_momentum_inertia_to_nodes(
    mpm::VelocityUpdate velocity_update) noexcept {
  // Map mass and momentum to nodes
  this->map_mass_momentum_to_nodes(velocity_update);

  // Damping Functions
  const auto& Fe = this->evanescent_damping_functions();

  // Damping scalar
  double f_M = Fe.prod();

  // Map inertia to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_inertia(true, mpm::ParticlePhase::SinglePhase,
                              mass_ * shapefn_[i] * acceleration_ * f_M);
  }
}

//! Map damped mass vector to nodes
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_pml_properties_to_nodes() noexcept {
  // Check if particle mass is set
  assert(mass_ != std::numeric_limits<double>::max());

  // Recompute damping function at the beginning of time step
  this->compute_damping_functions(
      state_variables_[mpm::ParticlePhase::SinglePhase]);

  // Damping Functions
  const auto& Fe = this->evanescent_damping_functions();

  // Damping scalar
  Eigen::Matrix<double, Tdim, 1> f_M =
      Fe.prod() * Eigen::Matrix<double, Tdim, 1>::Ones();

  // Map damped mass, displacement vector to nodal property
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    const auto& damped_mass = mass_ * shapefn_[i] * f_M;
    nodes_[i]->update_property(true, "damped_masses", damped_mass, 0, Tdim);
    nodes_[i]->update_property(true, "damped_mass_displacements",
                               displacement_.cwiseProduct(damped_mass), 0,
                               Tdim);
    nodes_[i]->assign_pml(true);
  }
}

//! Map internal force
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_body_force(
    const VectorDim& pgravity) noexcept {
  // Damping functions
  const auto& Fe = this->evanescent_damping_functions();

  // Damping scalar
  double f_M = Fe.prod();

  // Compute nodal body forces
  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodes_[i]->update_external_force(true, mpm::ParticlePhase::SinglePhase,
                                     (pgravity * mass_ * f_M * shapefn_[i]));
}

//! Map internal force
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_internal_force(double dt) noexcept {
  // Call damping functions and pass for efficiency
  // Reduces number of normal_damping_function calls by 2. Evanescent included
  // for consistency
  const auto& Fe = this->evanescent_damping_functions();
  const auto& Fp = this->normal_damping_functions();

  // Compute internal force from total displacement
  this->map_internal_force_disp(Fp, dt);

  // Compute internal force from strain
  this->map_internal_force_strain(Fe, Fp, dt);

  // Compute internal force from stress
  this->map_internal_force_stress(Fp);

  // Compute internal force from stiffness
  this->map_internal_force_stiffness(dt);

  // Map anti-body force from initialized stress if any
  if (stress_.norm() > std::numeric_limits<double>::epsilon())
    this->map_anti_body_force(dt);
}

//! Compute internal force from total displacement
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_internal_force_disp(const Eigen::VectorXd& Fp,
                                                      double dt) noexcept {
  unsigned phase = mpm::ParticlePhase::SinglePhase;
  // Calculate damping value
  double f_H = Fp.prod();

  for (unsigned i = 0; i < nodes_.size(); ++i) {
    VectorDim nodal_force = -1.0 * nodes_[i]->previous_pml_displacement() *
                            shapefn_[i] * mass_ * f_H * dt;

    // Apply force to nodes
    nodes_[i]->update_internal_force(true, phase, nodal_force);
  }
}

//! Compute internal force from strain
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_internal_force_strain(
    const Eigen::VectorXd& Fe, const Eigen::VectorXd& Fp, double dt) noexcept {
  // Call reduced material and strain information
  const auto& de = this->reduce_dmatrix(this->constitutive_matrix_);
  auto strain = this->reduce_voigt(this->strain_);
  auto strain_int = this->reduce_voigt(this->call_state_var(true));

  // Call and compute damping functions
  const auto& Fee = this->combined_damping_functions(Fe, Fe, true);
  const auto& Fep = this->combined_damping_functions(Fe, Fp, false);
  const auto& Fpp = this->combined_damping_functions(Fp, Fp, true);
  const auto& Fl = (Fe / dt + Fp).cwiseInverse();
  const auto& Feps = this->voigt_damping_functions(Fe.cwiseProduct(Fl));
  const auto& Frho = this->voigt_damping_functions(Fp.cwiseProduct(Fl));

  // Construct nodal B matrices
  const Eigen::MatrixXd& Bee = (this->compute_damped_bmatrix(Fee)).transpose();
  const Eigen::MatrixXd& Bep = (this->compute_damped_bmatrix(Fep)).transpose();
  const Eigen::MatrixXd& Bpp = (this->compute_damped_bmatrix(Fpp)).transpose();
  const Eigen::MatrixXd& Bt = Bep + Bpp * dt;
  const Eigen::MatrixXd& Btt = Bee + Bt * dt;

  // Calculate force
  const Eigen::VectorXd& force =
      (Btt * de * Feps.matrix().asDiagonal() / dt / dt) * strain +
      (Bt * de - Btt * de * Frho.matrix().asDiagonal() / dt) * strain_int;

  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Segment nodal force
    VectorDim nodal_force = -1.0 * force.segment(i * Tdim, Tdim) * volume_;

    // Update nodes
    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::SinglePhase,
                                     nodal_force);
  }
}

//! Compute internal force from stress
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_internal_force_stress(
    const Eigen::VectorXd& Fp) noexcept {
  // Call reduced stress information
  auto stress_2int = this->reduce_voigt(this->call_state_var(false));

  // Compute damping functions
  const auto& Fpp = this->combined_damping_functions(Fp, Fp, true);

  // Construct nodal B matrices
  const Eigen::MatrixXd& Bpp = (this->compute_damped_bmatrix(Fpp)).transpose();

  // Calculate force
  const Eigen::VectorXd& force = Bpp * stress_2int;

  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Segment nodal force
    VectorDim nodal_force = -1.0 * force.segment(i * Tdim, Tdim) * volume_;

    // Update nodes
    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::SinglePhase,
                                     nodal_force);
  }
}

//! Compute stiffness internal force from stiffness
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_internal_force_stiffness(double dt) noexcept {
  unsigned phase = mpm::ParticlePhase::SinglePhase;
  // Compute local stiffness matrix
  Eigen::MatrixXd local_stiffness = this->compute_pml_stiffness_matrix(dt);

  // Displacement vector
  Eigen::VectorXd disp(Tdim * nodes_.size());
  disp.setZero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    disp.segment(i * Tdim, Tdim) +=
        nodes_[i]->previous_pml_displacement() + nodes_[i]->displacement(phase);
  }

  // Compute force
  Eigen::VectorXd force = -1.0 * local_stiffness * disp * volume_;
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Segment nodal force
    VectorDim nodal_force = force.segment(i * Tdim, Tdim);

    // Update nodes
    nodes_[i]->update_internal_force(true, phase, nodal_force);
  }
}

// Compute internal force from damping
//! Map PML rayleigh damping force
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_rayleigh_damping_force(double damping_factor,
                                                         double dt) noexcept {
  // Compute local damping matrix
  Eigen::MatrixXd damping_matrix = this->compute_pml_damping_matrix(dt);
  // Velocity vector
  Eigen::VectorXd vel(Tdim * nodes_.size());
  vel.setZero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    vel.segment(i * Tdim, Tdim) +=
        nodes_[i]->velocity(mpm::ParticlePhase::SinglePhase);
  }

  // Compute force
  Eigen::VectorXd force = -1.0 * damping_matrix * vel * volume_;
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Segment nodal force
    VectorDim nodal_force = force.segment(i * Tdim, Tdim);

    // Update nodes
    nodes_[i]->update_external_force(true, mpm::ParticlePhase::SinglePhase,
                                     nodal_force);
  }
}

//! Compute inertial force
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::map_inertial_force(double bossak_alpha) noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);

  // Damping Functions
  const auto& Fe = this->evanescent_damping_functions();

  // Damping scalar
  double f_M = Fe.prod();

  // Compute nodal inertial forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Modify nodal acceleration
    const VectorDim& nodal_acceleration =
        nodes_[i]->acceleration(mpm::ParticlePhase::SinglePhase);
    VectorDim nodal_force =
        -1.0 * nodal_acceleration * mass_ * shapefn_[i] * f_M;
    nodes_[i]->update_external_force(true, mpm::ParticlePhase::SinglePhase,
                                     nodal_force);
  }
}

//! Map material stiffness matrix to cell (used in equilibrium equation LHS)
template <unsigned Tdim>
inline bool mpm::ParticleUPML<Tdim>::map_material_stiffness_matrix_to_cell(
    double dt) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Compute local stiffness matrix
    Eigen::MatrixXd local_stiffness = this->compute_pml_stiffness_matrix(dt);

    // Compute local material stiffness matrix
    cell_->compute_local_stiffness_matrix_block(0, 0, local_stiffness, volume_,
                                                1.0);
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map mass matrix to cell (used in poisson equation LHS)
template <unsigned Tdim>
inline bool mpm::ParticleUPML<Tdim>::map_mass_matrix_to_cell(
    double newmark_beta, double bossak_alpha, double dt) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Damping functions
    const auto& Fe = this->evanescent_damping_functions();

    // Damping Scalar
    double f_M = Fe.prod();

    // Modify mass density
    double mass_density_mod = f_M * mass_density_;

    // Construct matrix
    Eigen::MatrixXd mass_density_matrix(Tdim * nodes_.size(),
                                        Tdim * nodes_.size());
    mass_density_matrix.setZero();
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      for (unsigned k = 0; k < Tdim; ++k) {
        mass_density_matrix(Tdim * i + k, Tdim * i + k) +=
            shapefn_[i] * mass_density_mod;
      }
    }

    // Compute local mass matrix
    cell_->compute_local_stiffness_matrix_block(
        0, 0, mass_density_matrix, volume_, 1.0 / (newmark_beta * dt * dt));
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map PML rayleigh damping matrix to cell (used in equilibrium
//! equation LHS)damping_matrix
template <unsigned Tdim>
inline bool mpm::ParticleUPML<Tdim>::map_rayleigh_damping_matrix_to_cell(
    double newmark_gamma, double newmark_beta, double dt,
    double damping_factor) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Compute local damping matrix
    Eigen::MatrixXd damping_matrix = this->compute_pml_damping_matrix(dt);

    // Compute local mass matrix
    cell_->compute_local_stiffness_matrix_block(
        0, 0, damping_matrix, volume_, newmark_gamma / (newmark_beta * dt));
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

template <unsigned Tdim>
inline Eigen::MatrixXd mpm::ParticleUPML<Tdim>::compute_pml_damping_matrix(
    double dt) noexcept {
  // Initialise
  Eigen::MatrixXd damping_matrix(Tdim * nodes_.size(), Tdim * nodes_.size());
  damping_matrix.setZero();

  // Reduce constitutive matrix
  const auto& de = this->reduce_dmatrix(this->constitutive_matrix_);

  // Damping functions
  const auto& Fp = this->normal_damping_functions();
  const auto& Fe = this->evanescent_damping_functions();
  const auto& Fl = (Fe / dt + Fp).cwiseInverse();
  const auto& Feps = Fe.cwiseProduct(Fl);
  const auto& Fee = this->combined_damping_functions(Fe, Fe, true);
  const auto& Fep = this->combined_damping_functions(Fe, Fp, false);
  const auto& Fpp = this->combined_damping_functions(Fp, Fp, true);

  // Damping scalar
  double const& f_C =
      Fe(0) * Fe(1) * Fp(2) + Fe(0) * Fe(2) * Fp(1) + Fe(1) * Fe(2) * Fp(0);

  // Construct nodal B matrices
  const Eigen::MatrixXd& Bee = this->compute_damped_bmatrix(Fee);
  const Eigen::MatrixXd& Bep = this->compute_damped_bmatrix(Fep);
  const Eigen::MatrixXd& Bpp = this->compute_damped_bmatrix(Fpp);
  const Eigen::MatrixXd& Btt = Bee + Bep * dt + Bpp * dt * dt;
  const Eigen::MatrixXd& Beps = this->compute_combined_bmatrix(Feps, Fl);

  // Modified mass density
  double mass_density_mod = mass_density_ * f_C;

  // Construct matrix
  Eigen::MatrixXd mass_density_matrix(Tdim * nodes_.size(),
                                      Tdim * nodes_.size());
  mass_density_matrix.setZero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    for (unsigned k = 0; k < Tdim; ++k) {
      mass_density_matrix(Tdim * i + k, Tdim * i + k) +=
          shapefn_[i] * mass_density_mod;
    }
  }

  // Compute damping matrix
  damping_matrix += mass_density_matrix + Btt.transpose() * de * Beps / dt;

  return damping_matrix;
}

//! Compute PML stiffness matrix
template <unsigned Tdim>
inline Eigen::MatrixXd mpm::ParticleUPML<Tdim>::compute_pml_stiffness_matrix(
    double dt) noexcept {
  // Initialise
  Eigen::MatrixXd local_stiffness(Tdim * nodes_.size(), Tdim * nodes_.size());
  local_stiffness.setZero();

  // Damping matrices
  const auto& Fp = this->normal_damping_functions();
  const auto& Fe = this->evanescent_damping_functions();
  const auto& Fl = (Fe / dt + Fp).cwiseInverse();
  const auto& Frho = Fp.cwiseProduct(Fl);
  const auto& Fee = this->combined_damping_functions(Fe, Fe, true);
  const auto& Fep = this->combined_damping_functions(Fe, Fp, false);
  const auto& Fpp = this->combined_damping_functions(Fp, Fp, true);

  // Damping Constants
  double f_K =
      Fp(0) * Fp(1) * Fe(2) + Fp(0) * Fp(2) * Fe(1) + Fp(1) * Fp(2) * Fe(0);
  double f_H = Fp.prod();

  // Call constitutive matrix
  const auto& de = this->reduce_dmatrix(this->constitutive_matrix_);

  // Construct nodal B matrices
  const Eigen::MatrixXd& Bee = (this->compute_damped_bmatrix(Fee)).transpose();
  const Eigen::MatrixXd& Bep = (this->compute_damped_bmatrix(Fep)).transpose();
  const Eigen::MatrixXd& Bpp = (this->compute_damped_bmatrix(Fpp)).transpose();
  const Eigen::MatrixXd& Btt = Bee + Bep * dt + Bpp * dt * dt;
  const Eigen::MatrixXd& Brho = this->compute_combined_bmatrix(Frho, Fl);

  double mass_density_mod = mass_density_ * (f_K + dt * f_H);
  // Construct matrix
  Eigen::MatrixXd mass_density_matrix(Tdim * nodes_.size(),
                                      Tdim * nodes_.size());
  mass_density_matrix.setZero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    for (unsigned k = 0; k < Tdim; ++k) {
      mass_density_matrix(Tdim * i + k, Tdim * i + k) +=
          shapefn_[i] * mass_density_mod;
    }
  }

  // Calculate local stiffness matrix
  local_stiffness += mass_density_matrix + Btt * de * Brho / dt;

  return local_stiffness;
}

// Compute strain increment of PML Particle
template <unsigned Tdim>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticleUPML<Tdim>::compute_strain_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase, double dt) noexcept {
  // Damping matrices
  const auto& Fp = this->normal_damping_functions();
  const auto& Fe = this->evanescent_damping_functions();
  const auto& Fl = (Fe / dt + Fp).cwiseInverse();
  const auto& Frho = Fp.cwiseProduct(Fl);
  const auto& Feps = Fe.cwiseProduct(Fl);

  // Call reduced strain values
  auto strain_int = this->reduce_voigt(this->call_state_var(true));
  auto strain = this->reduce_voigt(this->strain_);

  // Construct nodal B matrices
  const Eigen::MatrixXd& Brho = this->compute_combined_bmatrix(Frho, Fl);
  const Eigen::MatrixXd& Beps = this->compute_combined_bmatrix(Feps, Fl);

  // Adjust voigt
  const auto& Feps_voigt = this->voigt_damping_functions(Feps);
  const auto& Frho_voigt = this->voigt_damping_functions(Frho);

  // Velocity vector
  Eigen::VectorXd vel(Tdim * nodes_.size());
  vel.setZero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    vel.segment(i * Tdim, Tdim) += nodes_[i]->velocity(phase);
  }

  // Displacement vector
  Eigen::VectorXd disp(Tdim * nodes_.size());
  disp.setZero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    disp.segment(i * Tdim, Tdim) +=
        nodes_[i]->previous_pml_displacement() + nodes_[i]->displacement(phase);
  }

  // Compute new strain
  auto reduced_dstrain = (Beps * vel + Brho * disp +
                          Feps_voigt.matrix().asDiagonal() * strain / dt -
                          Frho_voigt.matrix().asDiagonal() * strain_int) /
                             dt -
                         strain;

  // Apply strain to proper dimension
  Eigen::Matrix<double, 6, 1> dstrain;
  dstrain.setZero();
  switch (Tdim) {
    case 1:
      dstrain(0, 0) += reduced_dstrain(0, 0);
      break;
    case 2:
      dstrain(0, 0) += reduced_dstrain(0, 0);
      dstrain(1, 0) += reduced_dstrain(1, 0);
      dstrain(3, 0) += reduced_dstrain(2, 0);
      break;
    case 3:
      dstrain += reduced_dstrain;
      break;
  }
  for (unsigned i = 0; i < dstrain.size(); ++i)
    if (std::fabs(dstrain[i]) < 1.E-15) dstrain[i] = 0.;
  return dstrain;
}

//! Function to update pml displacement functions
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::update_pml_properties(double dt) noexcept {
  unsigned phase = mpm::ParticlePhase::SinglePhase;

  // Call PML state variables
  auto strain_int = this->call_state_var(true);
  auto stress_2int = this->call_state_var(false);

  // Calculate updated values
  Eigen::Matrix<double, 6, 1> new_strain_int = strain_int + this->strain_ * dt;
  Eigen::Matrix<double, 6, 1> new_stress_2int =
      stress_2int + this->constitutive_matrix_ * new_strain_int * dt;

  // Update PML state variable
  state_variables_[phase]["strain_int_xx"] = new_strain_int[0];
  state_variables_[phase]["strain_int_yy"] = new_strain_int[1];
  state_variables_[phase]["strain_int_zz"] = new_strain_int[2];
  state_variables_[phase]["strain_int_xy"] = new_strain_int[3];
  state_variables_[phase]["strain_int_yz"] = new_strain_int[4];
  state_variables_[phase]["strain_int_xz"] = new_strain_int[5];

  state_variables_[phase]["stress_2int_xx"] = new_stress_2int[0];
  state_variables_[phase]["stress_2int_yy"] = new_stress_2int[1];
  state_variables_[phase]["stress_2int_zz"] = new_stress_2int[2];
  state_variables_[phase]["stress_2int_xy"] = new_stress_2int[3];
  state_variables_[phase]["stress_2int_yz"] = new_stress_2int[4];
  state_variables_[phase]["stress_2int_xz"] = new_stress_2int[5];
}

//! Initialise damping functions
template <unsigned Tdim>
void mpm::ParticleUPML<Tdim>::compute_damping_functions(
    mpm::dense_map& state_vars) noexcept {
  const double& dpower =
      (this->material())
          ->template property<double>(std::string("damping_power"));
  const double& boundary_thickness = state_vars.at("boundary_thickness");

  const double& multiplier = std::pow(1.0 / boundary_thickness, dpower);

  // Compute damping functions in state variables
  state_vars.at("damping_function_x") =
      multiplier * std::pow(state_vars.at("distance_function_x"), dpower);
  state_vars.at("damping_function_y") =
      multiplier * std::pow(state_vars.at("distance_function_y"), dpower);
  state_vars.at("damping_function_z") =
      multiplier * std::pow(state_vars.at("distance_function_z"), dpower);
}

//! Function to return damping factors for waves propagating normal to boundary
template <unsigned Tdim>
Eigen::VectorXd mpm::ParticleUPML<Tdim>::normal_damping_functions()
    const noexcept {
  Eigen::Vector3d damping_functions;
  // PML material properties
  const double& beta =
      (this->material())
          ->template property<double>(std::string("maximum_damping_ratio"));

  // Damping functions
  damping_functions(0) =
      beta * state_variables_[mpm::ParticlePhase::SinglePhase].at(
                 "damping_function_x");
  damping_functions(1) =
      beta * state_variables_[mpm::ParticlePhase::SinglePhase].at(
                 "damping_function_y");
  damping_functions(2) =
      beta * state_variables_[mpm::ParticlePhase::SinglePhase].at(
                 "damping_function_z");

  return damping_functions;
}

//! Function to return damping factors for evanescent waves
template <unsigned Tdim>
Eigen::VectorXd mpm::ParticleUPML<Tdim>::evanescent_damping_functions()
    const noexcept {
  // Call necessary PML variables
  const double& beta =
      (this->material())
          ->template property<double>(std::string("maximum_damping_ratio"));
  const double& h =
      (this->material())
          ->template property<double>(std::string("characteristic_length"));

  // If no normal damping, calculate evanescent damping as h = alpha
  if (beta == 0) {
    Eigen::Vector3d damping_functions;
    // Damping functions
    damping_functions(0) =
        1.0 + h * state_variables_[mpm::ParticlePhase::SinglePhase].at(
                      "damping_function_x");
    damping_functions(1) =
        1.0 + h * state_variables_[mpm::ParticlePhase::SinglePhase].at(
                      "damping_function_y");
    damping_functions(2) =
        1.0 + h * state_variables_[mpm::ParticlePhase::SinglePhase].at(
                      "damping_function_z");

    return damping_functions;
  } else {
    // Call normal damping functions
    Eigen::VectorXd damping_functions = this->normal_damping_functions();

    // Call Material constants
    const double& E =
        (this->material())
            ->template property<double>(std::string("youngs_modulus"));
    const double& nu =
        (this->material())
            ->template property<double>(std::string("poisson_ratio"));
    const double& density =
        (this->material())->template property<double>(std::string("density"));
    const double& lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
    const double& shear_modulus = E / (2.0 * (1.0 + nu));
    const double& c_p = std::pow((lambda + 2. * shear_modulus) / density, 0.5);

    // Commpute evanescent damping functions
    damping_functions =
        Eigen::VectorXd::Constant(3, 1.0) + damping_functions * h / c_p;

    return damping_functions;
  }
}

// Compute 1D B matrix
template <>
inline Eigen::MatrixXd mpm::ParticleUPML<1>::compute_damped_bmatrix(
    const Eigen::VectorXd& damping_functions) noexcept {
  // Initialise B matrix
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(1, this->nodes_.size());
  bmatrix.setZero();

  // Incorporate shape function gradients and damping
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, i) = damping_functions(0) * dn_dx_(i, 0);
  }
  return bmatrix;
}

// Compute 2D B matrix
template <>
inline Eigen::MatrixXd mpm::ParticleUPML<2>::compute_damped_bmatrix(
    const Eigen::VectorXd& damping_functions) noexcept {
  // Initialise B matrix
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(3, 2 * this->nodes_.size());
  bmatrix.setZero();

  // Incorporate shape function gradients and damping
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, 2 * i) = damping_functions(0) * dn_dx_(i, 0);
    bmatrix(2, 2 * i) = damping_functions(1) * dn_dx_(i, 1);

    bmatrix(1, 2 * i + 1) = damping_functions(1) * dn_dx_(i, 1);
    bmatrix(2, 2 * i + 1) = damping_functions(0) * dn_dx_(i, 0);
  }
  return bmatrix;
}

// Compute 3D B matrix
template <>
inline Eigen::MatrixXd mpm::ParticleUPML<3>::compute_damped_bmatrix(
    const Eigen::VectorXd& damping_functions) noexcept {
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(6, 3 * this->nodes_.size());
  bmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, 3 * i) = damping_functions(0) * dn_dx_(i, 0);
    bmatrix(3, 3 * i) = damping_functions(1) * dn_dx_(i, 1);
    bmatrix(4, 3 * i) = damping_functions(2) * dn_dx_(i, 2);

    bmatrix(1, 3 * i + 1) = damping_functions(1) * dn_dx_(i, 1);
    bmatrix(3, 3 * i + 1) = damping_functions(0) * dn_dx_(i, 0);
    bmatrix(5, 3 * i + 1) = damping_functions(2) * dn_dx_(i, 2);

    bmatrix(2, 3 * i + 2) = damping_functions(2) * dn_dx_(i, 2);
    bmatrix(4, 3 * i + 2) = damping_functions(0) * dn_dx_(i, 0);
    bmatrix(5, 3 * i + 2) = damping_functions(1) * dn_dx_(i, 1);
  }
  return bmatrix;
}

// Compute 1D B matrix
template <>
inline Eigen::MatrixXd mpm::ParticleUPML<1>::compute_combined_bmatrix(
    const Eigen::VectorXd& Fa, const Eigen::VectorXd& Fb) noexcept {
  // Initialise B matrix
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(1, this->nodes_.size());
  bmatrix.setZero();

  // Incorporate shape function gradients and damping
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, i) = Fa(0) * Fb(0) * dn_dx_(i, 0);
  }
  return bmatrix;
}

// Compute 2D B matrix
template <>
inline Eigen::MatrixXd mpm::ParticleUPML<2>::compute_combined_bmatrix(
    const Eigen::VectorXd& Fa, const Eigen::VectorXd& Fb) noexcept {
  // Initialise B matrix
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(3, 2 * this->nodes_.size());
  bmatrix.setZero();

  // Incorporate shape function gradients and damping
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, 2 * i) = Fa(0) * Fb(0) * dn_dx_(i, 0);
    bmatrix(2, 2 * i) = Fa(0) * Fb(1) * dn_dx_(i, 1);

    bmatrix(1, 2 * i + 1) = Fa(1) * Fb(1) * dn_dx_(i, 1);
    bmatrix(2, 2 * i + 1) = Fa(1) * Fb(0) * dn_dx_(i, 0);
  }
  return bmatrix;
}

// Compute 3D B matrix
template <>
inline Eigen::MatrixXd mpm::ParticleUPML<3>::compute_combined_bmatrix(
    const Eigen::VectorXd& Fa, const Eigen::VectorXd& Fb) noexcept {
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(6, 3 * this->nodes_.size());
  bmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, 3 * i) = Fa(0) * Fb(0) * dn_dx_(i, 0);
    bmatrix(3, 3 * i) = Fa(0) * Fb(1) * dn_dx_(i, 1);
    bmatrix(4, 3 * i) = Fa(0) * Fb(2) * dn_dx_(i, 2);

    bmatrix(1, 3 * i + 1) = Fa(1) * Fb(1) * dn_dx_(i, 1);
    bmatrix(3, 3 * i + 1) = Fa(1) * Fb(0) * dn_dx_(i, 0);
    bmatrix(5, 3 * i + 1) = Fa(1) * Fb(2) * dn_dx_(i, 2);

    bmatrix(2, 3 * i + 2) = Fa(2) * Fb(2) * dn_dx_(i, 2);
    bmatrix(4, 3 * i + 2) = Fa(2) * Fb(0) * dn_dx_(i, 0);
    bmatrix(5, 3 * i + 2) = Fa(2) * Fb(1) * dn_dx_(i, 1);
  }
  return bmatrix;
}

//! Function to return combined damping factors
template <unsigned Tdim>
Eigen::VectorXd mpm::ParticleUPML<Tdim>::combined_damping_functions(
    const Eigen::VectorXd& Fa, const Eigen::VectorXd& Fb,
    bool identical_index) const noexcept {
  // Initialise damping functions
  Eigen::Vector3d damping_functions;

  // Compute joint damping function
  if (identical_index) {
    damping_functions(0) = Fa(1) * Fb(2);
    damping_functions(1) = Fa(0) * Fb(2);
    damping_functions(2) = Fa(0) * Fb(1);
  } else {
    damping_functions(0) = Fa(1) * Fb(2) + Fb(1) * Fa(2);
    damping_functions(1) = Fa(0) * Fb(2) + Fb(0) * Fa(2);
    damping_functions(2) = Fa(0) * Fb(1) + Fb(0) * Fa(1);
  }

  return damping_functions;
}

template <>
Eigen::VectorXd mpm::ParticleUPML<1>::voigt_damping_functions(
    const Eigen::VectorXd& damping_functions) const noexcept {
  Eigen::VectorXd voigt_damping_functions(1);

  voigt_damping_functions(0) = std::pow(damping_functions(0), 2);

  return voigt_damping_functions;
}

template <>
Eigen::VectorXd mpm::ParticleUPML<2>::voigt_damping_functions(
    const Eigen::VectorXd& damping_functions) const noexcept {
  Eigen::VectorXd voigt_damping_functions(3);

  voigt_damping_functions(0) = std::pow(damping_functions(0), 2);
  voigt_damping_functions(1) = std::pow(damping_functions(1), 2);
  voigt_damping_functions(2) = damping_functions(0) * damping_functions(1);

  return voigt_damping_functions;
}

template <>
Eigen::VectorXd mpm::ParticleUPML<3>::voigt_damping_functions(
    const Eigen::VectorXd& damping_functions) const noexcept {
  Eigen::VectorXd voigt_damping_functions(6);

  voigt_damping_functions(0) = std::pow(damping_functions(0), 2);
  voigt_damping_functions(1) = std::pow(damping_functions(1), 2);
  voigt_damping_functions(2) = std::pow(damping_functions(2), 2);
  voigt_damping_functions(3) = damping_functions(0) * damping_functions(1);
  voigt_damping_functions(4) = damping_functions(1) * damping_functions(2);
  voigt_damping_functions(5) = damping_functions(0) * damping_functions(2);

  return voigt_damping_functions;
}

template <>
Eigen::MatrixXd mpm::ParticleUPML<1>::reduce_voigt(
    Eigen::Matrix<double, 6, 1> prop) noexcept {
  Eigen::Matrix<double, 1, 1> reduce_prop;

  reduce_prop(0) = prop(0);

  return reduce_prop;
}

template <>
Eigen::MatrixXd mpm::ParticleUPML<2>::reduce_voigt(
    Eigen::Matrix<double, 6, 1> prop) noexcept {
  Eigen::Matrix<double, 3, 1> reduce_prop;

  reduce_prop(0) = prop(0);
  reduce_prop(1) = prop(1);
  reduce_prop(2) = prop(3);

  return reduce_prop;
}

template <>
Eigen::MatrixXd mpm::ParticleUPML<3>::reduce_voigt(
    Eigen::Matrix<double, 6, 1> prop) noexcept {
  return prop;
}

template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::ParticleUPML<Tdim>::call_state_var(
    bool strain_bool) noexcept {
  Eigen::Matrix<double, 6, 1> full_time_int_var;
  unsigned phase = mpm::ParticlePhase::SinglePhase;

  if (strain_bool) {
    full_time_int_var[0] = state_variables_[phase]["strain_int_xx"];
    full_time_int_var[1] = state_variables_[phase]["strain_int_yy"];
    full_time_int_var[2] = state_variables_[phase]["strain_int_zz"];
    full_time_int_var[3] = state_variables_[phase]["strain_int_xy"];
    full_time_int_var[4] = state_variables_[phase]["strain_int_yz"];
    full_time_int_var[5] = state_variables_[phase]["strain_int_xz"];
  } else {
    full_time_int_var[0] = state_variables_[phase]["stress_2int_xx"];
    full_time_int_var[1] = state_variables_[phase]["stress_2int_yy"];
    full_time_int_var[2] = state_variables_[phase]["stress_2int_zz"];
    full_time_int_var[3] = state_variables_[phase]["stress_2int_xy"];
    full_time_int_var[4] = state_variables_[phase]["stress_2int_yz"];
    full_time_int_var[5] = state_variables_[phase]["stress_2int_xz"];
  }

  return full_time_int_var;
}