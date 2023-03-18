//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticlePML<Tdim>::ParticlePML(Index id, const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {
  // Logger
  std::string logger =
      "particle_pml" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticlePML<Tdim>::ParticlePML(Index id, const VectorDim& coord,
                                    bool status)
    : mpm::Particle<Tdim>(id, coord, status) {
  //! Logger
  std::string logger =
      "particle_pml" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Compute strain of the particle
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::compute_strain(double dt) noexcept {
  // Assign strain rate
  strain_rate_ = this->compute_strain_rate(dn_dx_, mpm::ParticlePhase::Solid);
  // Update dstrain
  dstrain_ = strain_rate_ * dt;
  // Update strain
  strain_.noalias() += dstrain_;

  // Compute at centroid
  // Strain rate for reduced integration
  const Eigen::Matrix<double, 6, 1> strain_rate_centroid =
      this->compute_strain_rate(dn_dx_centroid_, mpm::ParticlePhase::Solid);

  // Assign volumetric strain at centroid
  dvolumetric_strain_ = dt * strain_rate_centroid.head(Tdim).sum();
}

//! Map particle mass and momentum to nodes
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_mass_momentum_to_nodes(
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

      // Map mass and momentum to nodes
      for (unsigned i = 0; i < nodes_.size(); ++i) {
        // Map mass and momentum
        nodes_[i]->update_mass(true, mpm::ParticlePhase::Solid,
                               mass_ * shapefn_[i]);
        nodes_[i]->update_momentum(true, mpm::ParticlePhase::Solid,
                                   mass_ * shapefn_[i] * velocity_);
      }
      break;
  }
}

//! Map body force
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_body_force(
    const VectorDim& pgravity) noexcept {
  // Compute nodal body forces
  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodes_[i]->update_external_force(true, mpm::ParticlePhase::Solid,
                                     (pgravity * mass_ * shapefn_(i)));
}

//! Function to reinitialise material to be run at the beginning of each time
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::initialise_constitutive_law() noexcept {
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

//! Map particle mass, momentum and inertia to nodes
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_mass_momentum_inertia_to_nodes() noexcept {
  // Map mass and momentum to nodes
  this->map_mass_momentum_to_nodes();

  // Map inertia to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_inertia(true, mpm::ParticlePhase::Solid,
                              mass_ * shapefn_[i] * acceleration_);
  }
}

//! Map inertial force
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_inertial_force() noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);

  // Compute nodal inertial forces
  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodes_[i]->update_external_force(
        true, mpm::ParticlePhase::Solid,
        (-1. * nodes_[i]->acceleration(mpm::ParticlePhase::Solid) * mass_ *
         shapefn_(i)));
}

// Compute strain and volume of the particle using nodal displacement
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::compute_strain_volume_newmark() noexcept {
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

// Compute strain rate of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::ParticlePML<1>::compute_strain_rate(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 1, 1> vel = nodes_[i]->velocity(phase);
    strain_rate[0] += dn_dx(i, 0) * vel[0];
  }

  if (std::fabs(strain_rate(0)) < 1.E-15) strain_rate[0] = 0.;
  return strain_rate;
}

// Compute strain rate of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::ParticlePML<2>::compute_strain_rate(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 2, 1> vel = nodes_[i]->velocity(phase);
    strain_rate[0] += dn_dx(i, 0) * vel[0];
    strain_rate[1] += dn_dx(i, 1) * vel[1];
    strain_rate[3] += dn_dx(i, 1) * vel[0] + dn_dx(i, 0) * vel[1];
  }

  if (std::fabs(strain_rate[0]) < 1.E-15) strain_rate[0] = 0.;
  if (std::fabs(strain_rate[1]) < 1.E-15) strain_rate[1] = 0.;
  if (std::fabs(strain_rate[3]) < 1.E-15) strain_rate[3] = 0.;
  return strain_rate;
}

// Compute strain rate of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::ParticlePML<3>::compute_strain_rate(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 3, 1> vel = nodes_[i]->velocity(phase);
    strain_rate[0] += dn_dx(i, 0) * vel[0];
    strain_rate[1] += dn_dx(i, 1) * vel[1];
    strain_rate[2] += dn_dx(i, 2) * vel[2];
    strain_rate[3] += dn_dx(i, 1) * vel[0] + dn_dx(i, 0) * vel[1];
    strain_rate[4] += dn_dx(i, 2) * vel[1] + dn_dx(i, 1) * vel[2];
    strain_rate[5] += dn_dx(i, 2) * vel[0] + dn_dx(i, 0) * vel[2];
  }

  for (unsigned i = 0; i < strain_rate.size(); ++i)
    if (std::fabs(strain_rate[i]) < 1.E-15) strain_rate[i] = 0.;
  return strain_rate;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticlePML<1>::compute_strain_increment(const Eigen::MatrixXd& dn_dx,
                                                  unsigned phase) noexcept {
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
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticlePML<2>::compute_strain_increment(const Eigen::MatrixXd& dn_dx,
                                                  unsigned phase) noexcept {
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
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticlePML<3>::compute_strain_increment(const Eigen::MatrixXd& dn_dx,
                                                  unsigned phase) noexcept {
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

//! Map material stiffness matrix to cell (used in equilibrium equation LHS)
template <unsigned Tdim>
inline bool mpm::ParticlePML<Tdim>::map_material_stiffness_matrix_to_cell() {
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

//! Map mass matrix to cell (used in poisson equation LHS)
template <unsigned Tdim>
inline bool mpm::ParticlePML<Tdim>::map_mass_matrix_to_cell(double newmark_beta,
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